/**
 * Modulo API per Briscola AI
 * Gestisce tutta la comunicazione con il server backend
 */

const API = (() => {
    const STRINGS = {
        failedToCreateGame: 'Impossibile creare la partita',
        failedToGetGameState: 'Impossibile ottenere lo stato della partita',
        failedToPlayCard: 'Impossibile giocare la carta',
        failedToGetGameResult: 'Impossibile ottenere il risultato della partita',
        failedToAbandonGame: 'Impossibile abbandonare la partita',
        errorCreatingGame: 'Errore durante la creazione della partita',
        errorGettingGameState: 'Errore nel recupero dello stato della partita',
        errorPlayingCard: 'Errore durante la giocata della carta',
        errorGettingGameResult: 'Errore nel recupero del risultato della partita',
        errorAbandoningGame: 'Errore durante l’abbandono della partita',
        wsEstablished: 'Connessione WebSocket stabilita',
        wsErrorParsing: 'Errore nel parsing del messaggio WebSocket',
        wsError: 'Errore WebSocket',
        wsClosed: 'Connessione WebSocket chiusa',
        wsReconnecting: 'Tentativo di riconnessione WebSocket...',
        intentionalDisconnect: 'Disconnessione intenzionale',
    };

    const _requireServedOverHttp = () => {
        if (window.location.protocol === 'file:') {
            throw new Error(
                'La UI deve essere servita dal server (non aprire index.html da filesystem). Avvia `briscola-server` e visita http://localhost:8000.'
            );
        }
    };

    // Usa sempre la stessa origin da cui è servita la UI.
    const API_URL = new URL('/api', window.location.origin).toString().replace(/\/$/, '');

    // --- Avviso "server che si sveglia" (cold start cloud) ---------------------------
    //
    // Il deploy pubblico (FastAPI Cloud) usa lo scale-to-zero: dopo ~90s di inattività
    // la replica viene spenta e la prima richiesta successiva paga il risveglio
    // (~10-15s lato piattaforma). Senza feedback l'utente vede solo un bottone che
    // "non fa nulla" e pensa che il sito sia rotto.
    //
    // Strategia: ogni fetch REST parte con un timer; se la risposta non arriva entro
    // SLOW_REQUEST_NOTICE_MS notifichiamo il listener registrato (la UI mostra un
    // avviso non bloccante) e lo spegniamo appena la richiesta si conclude — sia in
    // caso di successo che di errore.
    //
    // Limite noto (documentato di proposito): il PRIMO caricamento assoluto della
    // pagina NON è coprebile da qui — il browser sta ancora aspettando l'HTML dal
    // server addormentato e questo JS non è ancora stato scaricato/eseguito. L'avviso
    // copre i casi reali successivi: creazione partita, azioni e fetch che colpiscono
    // una replica appena sveglia o in scale-up.
    //
    // Concorrenza: più richieste possono essere lente nello stesso momento (es. i
    // fetch di metadati all'avvio della home). Usiamo un CONTATORE, non un booleano:
    // l'avviso appare quando la prima richiesta sfora la soglia e sparisce solo
    // quando l'ULTIMA richiesta lenta si conclude.
    const SLOW_REQUEST_NOTICE_MS = 2500;
    let slowRequestListener = null;
    let slowRequestsActive = 0;

    /**
     * Registra il callback chiamato con `true` quando almeno una richiesta REST
     * supera la soglia di lentezza e con `false` quando non ce ne sono più.
     * Il layer API non conosce la UI: è game.js a collegare il listener.
     */
    const setSlowRequestListener = (listener) => {
        slowRequestListener = typeof listener === 'function' ? listener : null;
    };

    const _notifySlowRequests = (active) => {
        if (!slowRequestListener) return;
        try {
            slowRequestListener(active);
        } catch (error) {
            // Un bug nel listener UI non deve mai rompere la richiesta in corso.
            console.error('Errore nel listener slow-request:', error);
        }
    };

    /**
     * Wrapper di `fetch` (stessa firma) che segnala le richieste lente.
     * Da usare al posto di `fetch` per TUTTE le chiamate REST di questo modulo.
     */
    const _fetchWithWakeNotice = async (input, init) => {
        let countedAsSlow = false;
        const timerId = setTimeout(() => {
            countedAsSlow = true;
            slowRequestsActive += 1;
            if (slowRequestsActive === 1) _notifySlowRequests(true);
        }, SLOW_REQUEST_NOTICE_MS);
        try {
            return await fetch(input, init);
        } finally {
            clearTimeout(timerId);
            if (countedAsSlow) {
                slowRequestsActive -= 1;
                if (slowRequestsActive === 0) _notifySlowRequests(false);
            }
        }
    };

    let gameId = null;
    let playerIndex = null;
    let websocket = null;
    let pingIntervalId = null;
    let reconnectTimeoutId = null;
    let reconnectAttempt = 0;
    let intentionalDisconnect = false;
    let currentOnMessage = null;
    let currentCallbacks = null;
    // Liveness: timestamp dell'ultimo messaggio ricevuto (pong inclusi). Se il socket
    // resta "OPEN" ma muto oltre la soglia (connessione mezza morta: standby, cambio
    // rete), lo chiudiamo noi per innescare la riconnessione.
    let lastMessageAtMs = 0;
    const PING_INTERVAL_MS = 15000;
    const STALE_CONNECTION_MS = 45000;
    let lifecycleListenersRegistered = false;

    /**
     * Calcola un delay di riconnessione con backoff esponenziale (con jitter).
     *
     * Nota didattica:
     * - un retry "fisso" può martellare il server e creare burst di richieste
     * - un backoff riduce il carico e rende la UI più stabile in caso di rete ballerina
     */
    const _reconnectDelayMs = (attempt) => {
        const baseMs = 600;
        const maxMs = 10000;
        const factor = 1.6;
        const jitterPct = 0.2; // +/- 20%

        const exp = Math.min(maxMs, Math.round(baseMs * (factor ** Math.max(0, attempt - 1))));
        const jitter = exp * jitterPct * (Math.random() * 2 - 1);
        return Math.max(0, Math.round(exp + jitter));
    };

    const _closeActiveWebSocket = ({ resetGameInfo }) => {
        intentionalDisconnect = true;

        if (reconnectTimeoutId) {
            clearTimeout(reconnectTimeoutId);
            reconnectTimeoutId = null;
        }

        if (pingIntervalId) {
            clearInterval(pingIntervalId);
            pingIntervalId = null;
        }

        if (websocket) {
            // Stacca i gestori PRIMA di chiudere: `close()` scatena `onclose` in modo
            // asincrono, e un gestore rimasto vivo sul vecchio socket innescherebbe una
            // seconda catena di riconnessioni in parallelo alla nuova (flapping).
            websocket.onopen = null;
            websocket.onmessage = null;
            websocket.onerror = null;
            websocket.onclose = null;
            if (websocket.readyState !== WebSocket.CLOSED) {
                websocket.close(1000, STRINGS.intentionalDisconnect);
            }
        }
        websocket = null;

        if (resetGameInfo) {
            // Teardown completo (fine partita): qui sì che il backoff riparte da zero.
            reconnectAttempt = 0;
            API.gameId = null;
            API.playerIndex = null;
        }
    };

    /**
     * Forza un controllo di salute e, se serve, una riconnessione IMMEDIATA.
     *
     * Chiamato quando il browser segnala che le condizioni sono cambiate (rete tornata,
     * tab di nuovo visibile): inutile aspettare il timer di backoff se possiamo già
     * sapere che il socket è assente, chiuso o muto da troppo tempo.
     */
    const _kickReconnect = (reason) => {
        if (intentionalDisconnect || API.gameId === null || API.playerIndex === null) return;
        const socketAlive = websocket && websocket.readyState === WebSocket.OPEN;
        const stale = socketAlive && lastMessageAtMs > 0 && Date.now() - lastMessageAtMs > STALE_CONNECTION_MS;
        if (socketAlive && !stale) return;

        console.log(`WS health check (${reason}): riconnessione immediata`);
        if (reconnectTimeoutId) {
            clearTimeout(reconnectTimeoutId);
            reconnectTimeoutId = null;
        }
        connectWebSocket(API.gameId, API.playerIndex, currentCallbacks || currentOnMessage);
    };

    const _registerLifecycleListeners = () => {
        if (lifecycleListenersRegistered) return;
        lifecycleListenersRegistered = true;
        window.addEventListener('online', () => _kickReconnect('online'));
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible') _kickReconnect('visibilitychange');
        });
    };

    /**
     * Crea una nuova partita sul server
     * @param {Object} config - Configurazione della partita
     * @returns {Promise} - Promise con l'esito della creazione
     */
    const createGame = async (config) => {
        _requireServedOverHttp();
        try {
            const response = await _fetchWithWakeNotice(`${API_URL}/games`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || STRINGS.failedToCreateGame);
            }

            return await response.json();
        } catch (error) {
            console.error(`${STRINGS.errorCreatingGame}:`, error);
            throw error;
        }
    };

    /**
     * Elenca gli agenti IA disponibili (metadati per UI).
     *
     * Ritorna un oggetto del tipo:
     * {
     *   common_note_it: string,
     *   agents: [{ name: string, label: string, description_it: string }, ...]
     * }
     *
     * Nota compatibilità:
     * se il backend ritorna una lista (vecchio formato), normalizziamo a `{ agents, common_note_it: '' }`.
     */
    const getAiAgents = async () => {
        _requireServedOverHttp();
        try {
            const response = await _fetchWithWakeNotice(`${API_URL}/ai/agents`);

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Impossibile caricare la lista agenti IA');
            }

            const payload = await response.json();
            if (Array.isArray(payload)) {
                return { common_note_it: '', agents: payload };
            }
            return {
                common_note_it: payload?.common_note_it || '',
                agents: Array.isArray(payload?.agents) ? payload.agents : []
            };
        } catch (error) {
            console.error('Errore caricando agenti IA:', error);
            throw error;
        }
    };

    /**
     * Elenca i modelli locali `.npz` disponibili sul server (per l'agente `bc_model`).
     *
     * Ritorna un oggetto del tipo:
     * { models: [{ id, label, description_it, ... }, ...] }
     */
    const getAiModels = async () => {
        _requireServedOverHttp();
        try {
            const response = await _fetchWithWakeNotice(`${API_URL}/ai/models`);

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Impossibile caricare la lista modelli IA');
            }

            const payload = await response.json();
            return {
                recommended_model: payload?.recommended_model || '',
                models: Array.isArray(payload?.models) ? payload.models : []
            };
        } catch (error) {
            console.error('Errore caricando modelli IA:', error);
            throw error;
        }
    };

    /**
     * Metadati runtime del server (deploy/config).
     *
     * Esempio:
     * {
     *   code_version: string,
     *   rules_version: string,
     *   event_log_mode: 'debug'|'dataset'|'off',
     *   dataset_requires_consent: boolean
     * }
     */
    const getServerMeta = async () => {
        _requireServedOverHttp();
        try {
            const response = await _fetchWithWakeNotice(`${API_URL}/meta`);

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Impossibile caricare metadati server');
            }

            return await response.json();
        } catch (error) {
            console.error('Errore caricando metadati server:', error);
            throw error;
        }
    };

    /**
     * Ottiene lo stato corrente di una partita
     * @param {string} gameId - ID della partita
     * @param {number} playerIndex - Indice del giocatore per una vista specifica
     * @returns {Promise} - Promise con lo stato della partita
     */
    const getGameState = async (gameId, playerIndex) => {
        _requireServedOverHttp();
        try {
            const url = new URL(`${API_URL}/games/${gameId}`);
            if (playerIndex !== undefined) {
                url.searchParams.append('player_index', playerIndex);
            }

            const response = await _fetchWithWakeNotice(url);

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || STRINGS.failedToGetGameState);
            }

            return await response.json();
        } catch (error) {
            console.error(`${STRINGS.errorGettingGameState}:`, error);
            throw error;
        }
    };

    /**
     * Gioca una carta nella partita
     * @param {string} gameId - ID della partita
     * @param {number} playerIndex - Indice del giocatore
     * @param {number} cardIndex - Indice della carta nella mano del giocatore
     * @returns {Promise} - Promise con l'esito dell'azione
     */
    const playCard = async (gameId, playerIndex, cardIndex, clientMeta = null) => {
        _requireServedOverHttp();
        try {
            const payload = {
                game_id: gameId,
                player_index: playerIndex,
                card_index: cardIndex
            };
            if (clientMeta && typeof clientMeta === 'object') {
                if (Number.isFinite(clientMeta.observedServerVersion)) {
                    payload.client_observed_server_version = Math.trunc(clientMeta.observedServerVersion);
                }
                if (Number.isFinite(clientMeta.decisionTimeMs)) {
                    payload.client_decision_time_ms = Math.max(0, Math.trunc(clientMeta.decisionTimeMs));
                }
            }

            const response = await _fetchWithWakeNotice(`${API_URL}/games/${gameId}/actions`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || STRINGS.failedToPlayCard);
            }

            return await response.json();
        } catch (error) {
            console.error(`${STRINGS.errorPlayingCard}:`, error);
            throw error;
        }
    };

    /**
     * Ottiene il risultato finale di una partita
     * @param {string} gameId - ID della partita
     * @returns {Promise} - Promise con il risultato della partita
     */
    const getGameResult = async (gameId) => {
        _requireServedOverHttp();
        try {
            const response = await _fetchWithWakeNotice(`${API_URL}/games/${gameId}/result`);

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || STRINGS.failedToGetGameResult);
            }

            return await response.json();
        } catch (error) {
            console.error(`${STRINGS.errorGettingGameResult}:`, error);
            throw error;
        }
    };

    /**
     * Abbandona una partita in corso.
     *
     * Il backend rimuove la sessione e, se l'event log è attivo, la marca come abortita.
     * La UI tratta l'operazione come best-effort: anche se la rete cade, il giocatore può
     * comunque tornare alla schermata iniziale dopo la conferma.
     */
    const abandonGame = async (gameId, playerIndex) => {
        _requireServedOverHttp();
        try {
            const payload = {};
            if (Number.isInteger(playerIndex)) payload.player_index = playerIndex;
            const response = await _fetchWithWakeNotice(`${API_URL}/games/${gameId}/abandon`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || STRINGS.failedToAbandonGame);
            }

            return await response.json();
        } catch (error) {
            console.error(`${STRINGS.errorAbandoningGame}:`, error);
            throw error;
        }
    };

    /**
     * Connette al WebSocket per aggiornamenti in tempo reale
     * @param {string} gameId - ID della partita
     * @param {number} playerIndex - Indice del giocatore
     * @param {Function} onMessage - Callback per i messaggi ricevuti
     * @returns {WebSocket} - Connessione WebSocket
     */
    const connectWebSocket = (gameId, playerIndex, onMessage) => {
        _requireServedOverHttp();

        // Chiude un'eventuale connessione già esistente
        _closeActiveWebSocket({ resetGameInfo: false });

        // Determina l'URL del WebSocket (ws o wss in base a http/https), stessa origin della UI.
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsBase = `${wsProtocol}//${window.location.host}`;
        const wsUrl = new URL(`/api/ws/${gameId}/${playerIndex}`, wsBase).toString();

        intentionalDisconnect = false;
        currentOnMessage = onMessage;
        currentCallbacks = null;

        // Backward compatibility: se il terzo argomento è un oggetto, lo trattiamo come callbacks.
        if (onMessage && typeof onMessage === 'object') {
            currentCallbacks = onMessage;
            currentOnMessage = onMessage.onMessage;
        }

        _registerLifecycleListeners();
        const ws = new WebSocket(wsUrl);
        websocket = ws;
        lastMessageAtMs = Date.now();

        ws.onopen = () => {
            if (ws !== websocket) return; // istanza rimpiazzata nel frattempo
            console.log(STRINGS.wsEstablished);
            reconnectAttempt = 0;
            lastMessageAtMs = Date.now();
            // Salva ID partita e indice giocatore per la riconnessione
            API.gameId = gameId;
            API.playerIndex = playerIndex;

            if (reconnectTimeoutId) {
                clearTimeout(reconnectTimeoutId);
                reconnectTimeoutId = null;
            }

            if (currentCallbacks && typeof currentCallbacks.onOpen === 'function') {
                currentCallbacks.onOpen();
            }

            // Heartbeat: ping periodico + rilevazione delle connessioni "mezze morte".
            // Il server risponde pong, quindi su una connessione sana arriva traffico a ogni
            // giro; se il silenzio supera la soglia, chiudiamo NOI il socket cosi' `onclose`
            // innesca la normale catena di riconnessione (senza questo, dopo uno standby o
            // un cambio rete il socket puo' restare OPEN ma muto per sempre).
            if (pingIntervalId) {
                clearInterval(pingIntervalId);
                pingIntervalId = null;
            }
            pingIntervalId = setInterval(() => {
                if (ws !== websocket) return;
                if (ws.readyState !== WebSocket.OPEN) return;
                if (Date.now() - lastMessageAtMs > STALE_CONNECTION_MS) {
                    console.warn('WS muto oltre soglia: chiusura forzata per riconnettere');
                    ws.close(4000, 'stale connection');
                    return;
                }
                ws.send(JSON.stringify({ type: 'ping' }));
            }, PING_INTERVAL_MS);
        };

        ws.onmessage = (event) => {
            if (ws !== websocket) return; // messaggi tardivi di un socket rimpiazzato
            lastMessageAtMs = Date.now();
            try {
                const data = JSON.parse(event.data);

                // Keepalive: il backend risponde ai ping con `{type: "pong"}`.
                // Questi messaggi NON sono uno snapshot dello stato di gioco: se li passiamo
                // al layer UI rischiamo di “resettare” la mano/punti a valori vuoti e bloccare
                // la partita fino al prossimo evento.
                if (data && typeof data === 'object' && (data.type === 'pong' || data.type === 'ping')) {
                    return;
                }

                if (typeof currentOnMessage === 'function') {
                    currentOnMessage(data);
                }
            } catch (error) {
                console.error(`${STRINGS.wsErrorParsing}:`, error);
            }
        };

        ws.onerror = (error) => {
            if (ws !== websocket) return;
            console.error(`${STRINGS.wsError}:`, error);
            if (currentCallbacks && typeof currentCallbacks.onError === 'function') {
                currentCallbacks.onError(error);
            }
        };

        ws.onclose = (event) => {
            if (ws !== websocket) return; // chiusura di un socket gia' rimpiazzato: ignora
            console.log(`${STRINGS.wsClosed}:`, event.code, event.reason);

            if (pingIntervalId) {
                clearInterval(pingIntervalId);
                pingIntervalId = null;
            }

            if (currentCallbacks && typeof currentCallbacks.onClose === 'function') {
                currentCallbacks.onClose(event);
            }

            // Prova a riconnettersi dopo un ritardo se non è stata una chiusura intenzionale.
            if (!intentionalDisconnect) {
                reconnectAttempt += 1;
                const delayMs = _reconnectDelayMs(reconnectAttempt);

                if (currentCallbacks && typeof currentCallbacks.onReconnectAttempt === 'function') {
                    currentCallbacks.onReconnectAttempt({ attempt: reconnectAttempt, delayMs });
                }

                reconnectTimeoutId = setTimeout(() => {
                    if (API.gameId !== null && API.playerIndex !== null) {
                        console.log(STRINGS.wsReconnecting);
                        connectWebSocket(API.gameId, API.playerIndex, currentCallbacks || currentOnMessage);
                    }
                }, delayMs);
            }
        };

        return ws;
    };

    /**
     * Disconnette dal WebSocket
     */
    const disconnectWebSocket = () => {
        _closeActiveWebSocket({ resetGameInfo: true });
    };

    // API pubblica
    return {
        gameId,
        playerIndex,
        setSlowRequestListener,
        createGame,
        getAiAgents,
        getAiModels,
        getServerMeta,
        getGameState,
        playCard,
        getGameResult,
        abandonGame,
        connectWebSocket,
        disconnectWebSocket
    };
})();
