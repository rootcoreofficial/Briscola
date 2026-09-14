/**
 * Modulo di gioco per Briscola AI - Versione Semplificata
 *
 * Coordina la logica di gioco. La UI è ora guidata esclusivamente
 * dallo stato ricevuto via WebSocket dal backend.
 *
 * Modello "standard":
 * - il backend avanza automaticamente la partita (incluse le mosse IA)
 * - il frontend controlla solo la presentazione (hold/animazioni) degli update ricevuti.
 */

document.addEventListener('DOMContentLoaded', () => {
    /**
     * Durata (ms) dell'evidenziazione "reveal" prima di applicare gli update UI.
     *
     * Obiettivo didattico/UX:
     * - rendere percepibile la sequenza degli eventi (carta scelta -> carta sul tavolo)
     * - mantenere la stessa durata per player e IA (coerenza visiva)
     *
     * Nota: la vera sorgente di verità resta il backend; qui "tratteniamo" solo il rendering.
     */
    const REVEAL_DURATION_MS = 1400;
    const AI_PLAYER_DISPLAY_NAME = 'Giocatore AI';
    const STARTUP_MIN_VISIBLE_MS = 2500;
    const INITIAL_AI_START_MESSAGE_HOLD_MS = 1400;

    /**
     * Durata (ms) di visualizzazione del risultato della mano (chi vince + punti).
     *
     * Nota architetturale:
     * il backend invia `trick_result` e subito dopo anche uno snapshot aggiornato.
     * Per evitare che il risultato “sparisca” immediatamente, il frontend trattiene
     * lo snapshot finché non è passato questo tempo.
     */
    const TRICK_RESULT_HOLD_MS = 2000;

    /**
     * Identificatore pseudonimo del client (persistente in localStorage).
     *
     * Obiettivo:
     * - poter fare split train/val "per giocatore" senza salvare nomi o PII nel DB.
     * - avere un id stabile tra partite, ma non riconducibile a una persona.
     */
    const _getClientId = () => {
        try {
            const key = 'briscola_client_id';
            let value = window.localStorage.getItem(key);
            if (value && typeof value === 'string') return value;
            value = (window.crypto && typeof window.crypto.randomUUID === 'function')
                ? window.crypto.randomUUID()
                : `client_${Math.random().toString(16).slice(2)}_${Date.now()}`;
            window.localStorage.setItem(key, value);
            return value;
        } catch (e) {
            // Fallback: se localStorage non è disponibile, usiamo un id effimero.
            return `client_${Math.random().toString(16).slice(2)}_${Date.now()}`;
        }
    };

    // Game state - minimal, derived from backend
    const store = Store.create({
        gameId: null,
        playerName: null,
        playerIndex: 0,       // Human is always player 0
        opponentIndex: 1,     // AI is always player 1
        connected: false,
        observation: null,
        gameOver: false,
        actionInFlight: false
    });

    const getState = () => store.getState();

    // Metadati runtime del server (es. modalità raccolta dati e debug full-state).
    let serverMeta = { dataset_requires_consent: false, debug_state_endpoint_enabled: false };

    let aiCatalogsPromise = null;
    let aiCatalogsLoaded = false;
    let initialBoardReadyResolve = null;
    let initialBoardReadyTimeoutId = null;

    const loadRuntimeMetadata = async () => {
        try {
            const meta = await API.getServerMeta();
            serverMeta = meta && typeof meta === 'object' ? meta : serverMeta;
            const required = meta?.dataset_requires_consent === true;
            UI.setDataCollectionConsent({
                required,
                description_it: required
                    ? 'Questa istanza sta raccogliendo dataset umano: le mosse verranno registrate in modo anonimo.'
                    : ''
            });
        } catch (error) {
            // Se non riusciamo a caricare i meta, manteniamo la UI in modalità "no consent required".
            serverMeta = { dataset_requires_consent: false, debug_state_endpoint_enabled: false };
            UI.setDataCollectionConsent({ required: false, description_it: '' });
        }
    };

    const loadAiCatalogs = async () => {
        if (aiCatalogsLoaded) return;
        if (aiCatalogsPromise) return aiCatalogsPromise;

        UI.setAdvancedOptionsLoading(true);
        aiCatalogsPromise = (async () => {
            const [agentsResult, modelsResult] = await Promise.allSettled([
                API.getAiAgents(),
                API.getAiModels(),
            ]);

            if (agentsResult.status === 'fulfilled') {
                UI.setAiAgents(agentsResult.value);
            } else {
                console.warn('Impossibile caricare metadati agenti IA:', agentsResult.reason);
            }

            if (modelsResult.status === 'fulfilled') {
                UI.setAiModels(modelsResult.value);
            } else {
                console.warn('Impossibile caricare lista modelli IA (.npz):', modelsResult.reason);
                UI.setAiModels({ models: [] });
            }

            if (agentsResult.status === 'rejected' || modelsResult.status === 'rejected') {
                throw new Error('Impossibile caricare le opzioni IA. Riprova tra qualche secondo.');
            }
            aiCatalogsLoaded = true;
        })();

        try {
            await aiCatalogsPromise;
        } finally {
            aiCatalogsPromise = null;
            UI.setAdvancedOptionsLoading(false);
        }
    };

    // Track last known table state to detect changes
    let lastAppliedServerVersion = -1;
    let pollingIntervalId = null;
    let pollingInFlight = false;
    // Token di sessione polling: _stopPolling lo incrementa, così i pollOnce ancora in
    // volo di una sessione precedente non ri-schedulano nulla dopo lo stop.
    let pollingGeneration = 0;

    // Timing umano (client-side): stimiamo il tempo decisionale (ms) come il tempo trascorso
    // da quando la UI applica uno snapshot in cui `my_turn=true` fino al click.
    //
    // Nota:
    // - usiamo "apply time" (non "receive time") perché il frontend può trattenere snapshot per UX (hold).
    // - misuriamo solo quando parte un turno umano (transizione my_turn false -> true).
    let my_turn_started_at_ms = null;
    let my_turn_observation_server_version = null;
    let was_my_turn = false;

    // UI hold: quando evidenziamo una carta, rinviamo il rendering dello snapshot
    // finché non è passato il tempo di reveal (evita che la carta appaia sul tavolo
    // mentre è ancora "in mano").
    let uiHoldUntilMs = 0;
    let startupOverlayReleaseAtMs = 0;
    let startupEventsHoldUntilMs = 0;
    /**
     * Coda di eventi UI provenienti dal backend.
     *
     * Motivazione:
     * - Con il modello server-driven, gli eventi arrivano "subito" dal WS:
     *   snapshot, reveal IA, risultato mano, snapshot post-mano.
     * - Per mantenere una sequenza visiva didattica (carta 1 -> carta 2 -> risultato),
     *   il frontend mette in coda gli eventi e li consuma rispettando gli hold.
     *
     * Tipi attesi:
     * - { type: 'observation', data: <snapshot> }
     * - { type: 'ai_card_reveal', data: <message> }
     * - { type: 'trick_result', data: <message> }
     */
    let pendingEvents = [];
    let flushTimeoutId = null;
    let debugPeekActive = false;
    let debugPeekRequestId = 0;
    let activeOpponentReveal = null;
    let startingPlayerAnnounced = false;

    const _displayNameForPlayer = (playerIndex, fallbackName = null) => {
        if (playerIndex === getState().playerIndex) return 'Tu';
        if (playerIndex === getState().opponentIndex) return AI_PLAYER_DISPLAY_NAME;
        return fallbackName || `Giocatore ${playerIndex + 1}`;
    };

    const _normalizeResultDisplayNames = (result) => {
        if (!result || typeof result !== 'object' || result.is_team_game) return result;

        const state = getState();
        const obsPlayers = state.observation?.players || [];
        const humanName = obsPlayers.find(p => p.index === state.playerIndex)?.name || state.playerName || 'Tu';
        const aiBackendName = obsPlayers.find(p => p.index === state.opponentIndex)?.name || null;
        const normalized = { ...result };

        if (result.winner_index === state.opponentIndex) {
            normalized.winner = AI_PLAYER_DISPLAY_NAME;
        } else if (result.winner_index === state.playerIndex) {
            normalized.winner = humanName;
        }

        if (result.points && typeof result.points === 'object') {
            normalized.points = {};
            for (const [name, points] of Object.entries(result.points)) {
                const label = name === aiBackendName ? AI_PLAYER_DISPLAY_NAME : name;
                normalized.points[label] = points;
            }
        }
        return normalized;
    };

    /**
     * Modalità debug: fallback polling al posto del WebSocket.
     *
     * Attivazione:
     * - aggiungi `?polling=1` all'URL della UI (es. http://localhost:8000/?polling=1)
     *
     * Motivazione:
     * - utile quando stai debuggando problemi di rete/reconnect e vuoi un flusso più "semplice"
     * - non è pensato come modalità principale (il WS resta il path normale)
     */
    const _shouldUsePolling = () => {
        const params = new URLSearchParams(window.location.search);
        const polling = (params.get('polling') || '').toLowerCase();
        if (polling === '1' || polling === 'true') return true;
        if (polling === '0' || polling === 'false') return false;
        // Override esplicito verso WebSocket.
        const ws = (params.get('ws') || '').toLowerCase();
        if (ws === '1' || ws === 'true') return false;
        // Default deciso dal server: "polling" in cloud multi-replica, "ws" in locale.
        return (window.__BRISCOLA_REALTIME_MODE__ || 'ws') === 'polling';
    };

    const _stopPolling = () => {
        pollingGeneration += 1;
        if (pollingIntervalId) {
            clearTimeout(pollingIntervalId);
            pollingIntervalId = null;
        }
        pollingInFlight = false;
    };

    const _startPolling = (gameId, playerIndex) => {
        _stopPolling();
        UI.updateGameInfo({ connected: false, statusText: 'Polling', statusClass: 'connecting' });

        // Backoff sugli errori: il polling è un fallback di debug, ma a 700ms fissi un
        // backend irraggiungibile verrebbe martellato ~1.4 volte/secondo per tab aperta.
        // Dopo ogni errore l'intervallo raddoppia (fino a 5s) e si resetta al primo successo.
        const BASE_INTERVAL_MS = 700;
        const MAX_INTERVAL_MS = 5000;
        const generation = pollingGeneration;
        let currentIntervalMs = BASE_INTERVAL_MS;

        const scheduleNext = () => {
            if (generation !== pollingGeneration) return; // sessione fermata o rimpiazzata
            pollingIntervalId = setTimeout(pollOnce, currentIntervalMs);
        };

        const pollOnce = async () => {
            if (generation !== pollingGeneration) return;
            pollingInFlight = true;
            try {
                const obs = await API.getGameState(gameId, playerIndex);
                handleGameUpdate(obs);
                currentIntervalMs = BASE_INTERVAL_MS;
            } catch (error) {
                console.warn('Polling error:', error);
                currentIntervalMs = Math.min(currentIntervalMs * 2, MAX_INTERVAL_MS);
                UI.updateGameInfo({ connected: false, statusText: 'Polling: errore rete', statusClass: 'reconnecting' });
            } finally {
                pollingInFlight = false;
                scheduleNext();
            }
        };

        // Primo fetch immediato, poi catena di setTimeout (intervallo adattivo:
        // a differenza di setInterval, il prossimo giro parte solo a richiesta conclusa).
        pollingIntervalId = setTimeout(pollOnce, 0);
    };

    const _currentUiHoldUntilMs = () => Math.max(uiHoldUntilMs, startupEventsHoldUntilMs);

    const _scheduleFlush = () => {
        if (flushTimeoutId) {
            clearTimeout(flushTimeoutId);
            flushTimeoutId = null;
        }
        const delay = Math.max(0, _currentUiHoldUntilMs() - Date.now());
        flushTimeoutId = setTimeout(_flushPending, delay);
    };

    const _holdUiForReveal = () => {
        uiHoldUntilMs = Math.max(uiHoldUntilMs, Date.now() + REVEAL_DURATION_MS);
        _scheduleFlush();
    };

    const _holdUiForTrickResult = () => {
        uiHoldUntilMs = Math.max(uiHoldUntilMs, Date.now() + TRICK_RESULT_HOLD_MS);
        _scheduleFlush();
    };

    const _isTypingTarget = (target) => {
        const tag = target?.tagName?.toLowerCase();
        return tag === 'input' || tag === 'textarea' || tag === 'select' || target?.isContentEditable === true;
    };

    const _applyDebugPeekState = (fullState) => {
        if (!debugPeekActive || !fullState || fullState.type !== 'game_state') return;

        const state = getState();
        const opponent = (fullState.players || []).find((p) => p.index === state.opponentIndex);
        if (opponent?.hand) {
            UI.renderOpponentHand(opponent.hand_size || opponent.hand.length, opponent.hand);
        }
        UI.updateDeckCount(fullState.cards_remaining_in_deck || 0, fullState.next_deck_card || null);
    };

    const _restoreDebugPeekFromObservation = (obs) => {
        const state = getState();
        const opponent = (obs.players || []).find(p => p.index === state.opponentIndex);
        const opponentHandSize = opponent?.hand_size || 0;

        // Il debug peek sostituisce solo mano IA e mazzo: al rilascio di `S`
        // ripristiniamo quelle aree senza passare da updateUI(), così la mano
        // del giocatore non viene ricreata e non riparte l'animazione card-appear.
        UI.renderOpponentHand(opponentHandSize);
        if (
            activeOpponentReveal &&
            activeOpponentReveal.cardIndex >= 0 &&
            activeOpponentReveal.cardIndex < opponentHandSize
        ) {
            UI.revealOpponentCard(
                activeOpponentReveal.cardIndex,
                activeOpponentReveal.card,
                activeOpponentReveal.decisionType
            );
        }
        UI.updateDeckCount(obs.cards_remaining_in_deck || 0);
    };

    const _refreshDebugPeek = async () => {
        const state = getState();
        if (!debugPeekActive || !state.gameId || state.gameOver) return;

        const requestId = ++debugPeekRequestId;
        try {
            const fullState = await API.getGameState(state.gameId);
            if (!debugPeekActive || requestId !== debugPeekRequestId) return;
            _applyDebugPeekState(fullState);
        } catch (error) {
            console.warn('Debug peek failed:', error);
        }
    };

    const _setDebugPeekActive = (active) => {
        const next = active === true;
        if (debugPeekActive === next) return;

        if (next && serverMeta?.debug_state_endpoint_enabled !== true) {
            UI.showTurnMessage('Debug carte disabilitato su questo server');
            return;
        }

        debugPeekActive = next;
        debugPeekRequestId += 1;

        const state = getState();
        if (debugPeekActive) {
            _refreshDebugPeek();
        } else if (state.observation) {
            _restoreDebugPeekFromObservation(state.observation);
        }
    };

    const _startingPlayerMessage = (obs) => {
        if (startingPlayerAnnounced) return null;
        if (typeof obs?.first_player !== 'number') return null;

        const state = getState();
        if (obs.first_player === state.playerIndex) {
            return obs.my_turn ? 'Cominci tu - scegli una carta' : 'Hai cominciato tu';
        }
        if (obs.first_player === state.opponentIndex) {
            return obs.my_turn ? 'Ha cominciato l’IA - tocca a te' : 'Comincia l’IA...';
        }
        return `Comincia il giocatore ${obs.first_player + 1}`;
    };

    /**
     * Accoda un evento, collassando snapshot consecutivi.
     *
     * Gli snapshot (`observation`) sono ridondanti: se ne arrivano più di uno di fila
     * mentre siamo in hold, teniamo solo l'ultimo per evitare flicker e lavoro inutile.
     */
    const _enqueueEvent = (event) => {
        if (event.type === 'observation') {
            const last = pendingEvents[pendingEvents.length - 1];
            if (last?.type === 'observation') {
                pendingEvents[pendingEvents.length - 1] = event;
            } else {
                pendingEvents.push(event);
            }
        } else {
            pendingEvents.push(event);
        }
        _scheduleFlush();
    };

    const _applyObservation = (obs) => {
        // Guard rail: se arrivano snapshot fuori ordine (reconnect/ritardi), ignoriamo quelli vecchi.
        const serverVersion = typeof obs?.server_version === 'number' ? obs.server_version : -1;
        if (serverVersion !== -1 && serverVersion <= lastAppliedServerVersion) {
            console.warn('Ignoring stale observation:', { serverVersion, lastAppliedServerVersion, obs });
            return;
        }
        if (serverVersion !== -1) lastAppliedServerVersion = serverVersion;
        activeOpponentReveal = null;

        store.setState({
            observation: obs,
            gameOver: !!obs.game_over,
            // La UI è guidata dallo stato server: quando applichiamo uno snapshot valido,
            // possiamo considerare "chiusa" l'azione locale (lock click).
            actionInFlight: false
        });

        // Tracking decision time: segna inizio turno umano quando `my_turn` diventa true.
        // Non resettiamo se arrivano snapshot successivi nello stesso turno (evita di sottostimare).
        if (obs && typeof obs.my_turn === 'boolean') {
            if (obs.my_turn && !was_my_turn) {
                my_turn_started_at_ms = Date.now();
                my_turn_observation_server_version = typeof obs.server_version === 'number' ? obs.server_version : null;
            }
            was_my_turn = obs.my_turn;
        }

        updateUI(obs);

        if (
            initialBoardReadyResolve &&
            !obs.game_over &&
            obs.my_turn === false &&
            typeof obs.first_player === 'number' &&
            obs.first_player === getState().opponentIndex
        ) {
            const overlayReleaseAt = Math.max(Date.now(), startupOverlayReleaseAtMs);
            startupEventsHoldUntilMs = Math.max(
                startupEventsHoldUntilMs,
                overlayReleaseAt + INITIAL_AI_START_MESSAGE_HOLD_MS
            );
        }

        if (initialBoardReadyResolve) {
            UI.showGameBoard();
            const resolve = initialBoardReadyResolve;
            initialBoardReadyResolve = null;
            if (initialBoardReadyTimeoutId) {
                clearTimeout(initialBoardReadyTimeoutId);
                initialBoardReadyTimeoutId = null;
            }
            resolve(true);
        }

        if (obs.game_over) {
            handleGameOver();
        }
    };

    const _flushPending = () => {
        if (flushTimeoutId) {
            clearTimeout(flushTimeoutId);
            flushTimeoutId = null;
        }

        if (Date.now() < _currentUiHoldUntilMs()) {
            _scheduleFlush();
            return;
        }

        // Consuma quanti più eventi possibili finché non entriamo in un nuovo hold.
        while (pendingEvents.length > 0 && Date.now() >= _currentUiHoldUntilMs()) {
            const next = pendingEvents.shift();

            if (next.type === 'ai_card_reveal') {
                const data = next.data;
                activeOpponentReveal = {
                    cardIndex: data.card_index,
                    card: data.card,
                    decisionType: data.decision_type || null
                };
                UI.revealOpponentCard(data.card_index, data.card, data.decision_type || null);
                _holdUiForReveal();
                break;
            }

            if (next.type === 'trick_result') {
                handleTrickResult(next.data);
                break;
            }

            if (next.type === 'observation') {
                _applyObservation(next.data);
                continue;
            }

            // Tipo sconosciuto: logghiamo per debug e proseguiamo.
            console.warn('Evento WS con tipo non gestito:', next.type, next);
        }

        if (pendingEvents.length > 0) _scheduleFlush();
    };

    /**
     * Update the entire UI from observation
     */
    const updateUI = (obs) => {
        const state = getState();

        // Player hand
        const isMyTurn = obs.my_turn && !obs.game_over;
        UI.renderPlayerHand(obs.my_hand || [], isMyTurn, playCard);

        // Opponent hand and points (nuovo formato: array `players`)
        const opponent = (obs.players || []).find(p => p.index === state.opponentIndex);
        const opponentHandSize = opponent?.hand_size || 0;
        const opponentPoints = opponent?.points || 0;
        const opponentName = _displayNameForPlayer(state.opponentIndex, opponent?.name || 'Avversario IA');
        UI.renderOpponentHand(opponentHandSize);

        // Points
        UI.updatePlayerPoints(obs.my_points || 0);
        UI.updateOpponentInfo(opponentName, opponentPoints);

        // Table cards
        UI.renderTableCards(obs.table_cards || []);

        // Briscola: quando `trump_card` è null (es. deck vuoto) mostriamo comunque il seme.
        UI.renderTrumpCard(obs.trump_card, obs.trump_suit);

        // Deck count
        UI.updateDeckCount(obs.cards_remaining_in_deck || 0);

        if (debugPeekActive) {
            _refreshDebugPeek();
        }

        // Turn message
        const startingMessage = _startingPlayerMessage(obs);
        if (startingMessage) startingPlayerAnnounced = true;

        if (obs.game_over) {
            UI.showTurnMessage('Partita terminata');
        } else if (startingMessage) {
            UI.showTurnMessage(startingMessage, !obs.my_turn);
        } else if (obs.my_turn) {
            UI.showTurnMessage('Tocca a te - scegli una carta');
        } else {
            UI.showTurnMessage('Avversario sta pensando...', true);
        }

    };

    /**
     * Handle WebSocket messages
     */
    const handleGameUpdate = (data) => {
        // Ignore ping/pong
        if (data?.type === 'ping' || data?.type === 'pong') return;

        if (data?.type === 'ai_card_reveal') {
            _enqueueEvent({ type: 'ai_card_reveal', data });
            _flushPending();
            return;
        }

        if (data?.type === 'trick_result') {
            _enqueueEvent({ type: 'trick_result', data });
            _flushPending();
            return;
        }

        // Contratto WS: gli snapshot devono avere `type: "observation"`.
        if (data?.type !== 'observation') {
            console.warn('Unhandled WS message type:', data?.type, data);
            return;
        }

        // Validate it's an observation
        if (!Array.isArray(data.my_hand)) {
            console.warn('Ignoring invalid observation (no my_hand):', data);
            return;
        }

        _enqueueEvent({ type: 'observation', data });
        _flushPending();
    };

    /**
     * Handle trick result - display both cards and winner
     */
    const handleTrickResult = (data) => {
        const state = getState();
        activeOpponentReveal = null;

        // Render both cards on the table
        UI.renderTableCards(data.trick_cards || []);

        // Remove any revealed card from hand (to avoid duplication: card on table AND in hand)
        UI.removeRevealedCard();

        // Show winner message
        const winnerName = _displayNameForPlayer(data.winner_index, data.winner_name);
        const winnerLabel = data.winner_index === state.playerIndex ? 'Tu vinci!' : `${winnerName} vince!`;
        const pointsText = data.points > 0 ? ` (+${data.points} punti)` : '';
        UI.showTurnMessage(`${winnerLabel}${pointsText}`, false);

        // Trattieni la UI: lo snapshot “post mano” arriverà subito dopo, ma vogliamo
        // lasciare il tempo di leggere il risultato.
        _holdUiForTrickResult();
    };

    /**
     * Start a new game
     */
    const startGame = async (config) => {
        try {
            const aiAgent = config.aiAgent || 'random';
            const aiModelId = config.aiModelId || null;
            const aiModelCompatible = config.aiModelCompatible === true;
            const aiModelCompatibilityReasonIt = config.aiModelCompatibilityReasonIt || null;
            const agentRequiresModel =
                config.aiAgentRequiresModelSelection === true ||
                aiAgent === 'bc_model' ||
                aiAgent === 'bc_model_hybrid_endgame' ||
                aiAgent === 'bc_model_value_lookahead_8x8';

            // Il nome del player entra negli snapshot e nei messaggi di partita
            // (es. "X vince!"): teniamolo corto anche quando il modello selezionato
            // ha una label descrittiva molto lunga. La scelta dell'agente/modello resta
            // tracciata da `ai_agent` e `ai_model_id` nel payload.
            const playerNames = [config.playerName, AI_PLAYER_DISPLAY_NAME];

            if (serverMeta?.dataset_requires_consent === true && config.consentToDataCollection !== true) {
                throw new Error('Devi accettare la raccolta dati (anonima) per avviare la partita.');
            }

            if (agentRequiresModel && !aiModelId) {
                throw new Error('Seleziona un modello (.npz) prima di avviare la partita.');
            }
            if (agentRequiresModel && !aiModelCompatible) {
                const reason = aiModelCompatibilityReasonIt ? `\n\nMotivo: ${aiModelCompatibilityReasonIt}` : '';
                throw new Error(`Il modello selezionato non è compatibile.${reason}`);
            }

            const createPayload = {
                num_players: 2,
                player_names: playerNames,
                ai_agent: aiAgent,
                client_id: _getClientId(),
                consent_to_data_collection: config.consentToDataCollection === true,
            };
            if (agentRequiresModel) {
                createPayload.ai_model_id = aiModelId;
            }

            startupOverlayReleaseAtMs = Date.now() + STARTUP_MIN_VISIBLE_MS;
            startupEventsHoldUntilMs = 0;
            const startupMinimumVisible = new Promise((resolve) => setTimeout(resolve, STARTUP_MIN_VISIBLE_MS));
            const result = await API.createGame(createPayload);

            store.setState({
                gameId: result.game_id,
                playerName: config.playerName,
                playerIndex: 0,
                opponentIndex: 1,
                connected: false,
                observation: null,
                gameOver: false
            });

            my_turn_started_at_ms = null;
            my_turn_observation_server_version = null;
            was_my_turn = false;
            activeOpponentReveal = null;

            UI.setPlayerName(config.playerName);
            UI.updateGameInfo({ gameId: result.game_id, connected: false, statusText: 'Connessione...', statusClass: 'connecting' });

            // Assicura che le immagini delle carte siano in cache prima di mostrare il tavolo
            // (evita la comparsa "in ritardo" al primo render). Di norma il preload è già partito
            // alla home, quindi è istantaneo; il cap evita di bloccare l'avvio su reti lente.
            await Promise.race([
                UI.preloadCardAssets(),
                new Promise((resolve) => setTimeout(resolve, 3000)),
            ]);

            const firstRenderPromise = new Promise((resolve) => {
                initialBoardReadyResolve = resolve;
                initialBoardReadyTimeoutId = setTimeout(() => {
                    initialBoardReadyTimeoutId = null;
                    if (!initialBoardReadyResolve) return;
                    const resolveInitial = initialBoardReadyResolve;
                    initialBoardReadyResolve = null;
                    resolveInitial(false);
                }, 10000);
            });

            if (_shouldUsePolling()) {
                // Niente WS: solo polling (default in cloud multi-replica, o forzato via ?polling=1).
                _startPolling(result.game_id, 0);
            } else {
                // Connect WebSocket (path normale)
                API.connectWebSocket(result.game_id, 0, {
                    onMessage: handleGameUpdate,
                    onOpen: () => {
                        _stopPolling();
                        store.setState({ connected: true });
                        UI.updateGameInfo({ connected: true, statusText: 'Connesso', statusClass: 'connected' });
                    },
                    onClose: () => {
                        // Se la connessione cade durante un'azione, sblocchiamo la UI e ripristiniamo la mano.
                        const current = getState();
                        store.setState({ connected: false, actionInFlight: false });
                        UI.resetPlayerHandHighlights();
                        if (current.observation) updateUI(current.observation);

                        // Reset del buffer eventi: dopo reconnect useremo solo lo snapshot fresh dal server.
                        pendingEvents = [];
                        uiHoldUntilMs = 0;

                        UI.updateGameInfo({ connected: false, statusText: 'Non connesso', statusClass: 'disconnected' });
                    },
                    onReconnectAttempt: ({ attempt, delayMs }) => {
                        UI.updateGameInfo({
                            connected: false,
                            statusText: `Riconnessione... (tentativo ${attempt})`,
                            statusClass: 'reconnecting'
                        });
                        console.log(`WS reconnect attempt ${attempt} in ${delayMs}ms`);
                    }
                });
            }

            const renderedFromStream = await firstRenderPromise;
            if (!renderedFromStream) {
                const obs = await API.getGameState(result.game_id, 0);
                _applyObservation(obs);
                UI.showGameBoard();
            }

            await startupMinimumVisible;

        } catch (error) {
            UI.showSetupError(`Errore: ${error.message}`);
        }
    };

    /**
     * Risincronizza lo stato dalla verita' del server dopo un errore di rete.
     *
     * Robustezza: un blip di rete durante un'azione NON deve diventare subito un
     * errore in faccia al giocatore. Rileggiamo lo snapshot (GET idempotente, 2
     * tentativi con pausa crescente) e lasciamo decidere alla versione del server
     * se la mossa era arrivata oppure no.
     */
    const _resyncFromServer = async () => {
        for (const delayMs of [400, 1200]) {
            await new Promise((resolve) => setTimeout(resolve, delayMs));
            try {
                const current = getState();
                if (!current.gameId) return false;
                const obs = await API.getGameState(current.gameId, current.playerIndex);
                _applyObservation(obs);
                return true;
            } catch (error) {
                console.warn('Resync fallito, riprovo:', error?.message);
            }
        }
        return false;
    };

    /** Heuristica: errore di rete/transporto (fetch fallita) vs errore logico del server. */
    const _isNetworkError = (error) =>
        error instanceof TypeError || /fetch|network|load failed|connessione/i.test(error?.message || '');

    /**
     * Play a card
     */
    const playCard = async (cardIndex) => {
        const state = getState();
        if (!state.observation?.my_turn || state.gameOver || state.actionInFlight) return;

        try {
            // Feedback immediato: evidenziamo la carta scelta prima che venga "spostata"
            // sul tavolo tramite update WebSocket (effetto simile al reveal dell'IA).
            store.setState({ actionInFlight: true });
            UI.revealPlayerCard(cardIndex);
            _holdUiForReveal();

            const decisionTimeMs = my_turn_started_at_ms != null ? (Date.now() - my_turn_started_at_ms) : null;
            await API.playCard(state.gameId, state.playerIndex, cardIndex, {
                observedServerVersion: my_turn_observation_server_version,
                decisionTimeMs
            });
            // UI update will come via WebSocket
        } catch (error) {
            // In caso di errore, sblocchiamo la UI: lo snapshot potrebbe non arrivare.
            store.setState({ actionInFlight: false });

            // Blip di rete: prima di mostrare un errore, risincronizza col server e
            // decidi in base alla versione se la mossa era passata (risposta persa)
            // oppure no (richiesta persa). Solo se anche il resync fallisce mostriamo
            // l'errore: a quel punto il problema e' reale, non un singhiozzo.
            const versionBefore = typeof state.observation?.server_version === 'number'
                ? state.observation.server_version
                : null;
            if (_isNetworkError(error)) {
                UI.showTurnMessage('Connessione instabile: sincronizzo con il server...');
                const recovered = await _resyncFromServer();
                if (recovered) {
                    const obs = getState().observation;
                    const versionAfter = typeof obs?.server_version === 'number' ? obs.server_version : null;
                    if (versionBefore !== null && versionAfter !== null && versionAfter > versionBefore) {
                        // La mossa era arrivata (si e' persa solo la risposta): UI gia' aggiornata.
                        UI.showTurnMessage('');
                    } else {
                        // La mossa non e' arrivata al server: ripristina la mano e invita a riprovare.
                        if (obs) updateUI(obs);
                        UI.showTurnMessage('La mossa non è arrivata al server: riprova.');
                    }
                    return;
                }
            }

            // Ripristina la mano "normale" (rimuove highlight/disabled) ri-renderizzando dallo stato corrente.
            if (state.observation) updateUI(state.observation);
            UI.showTurnMessage(`Errore: ${error.message}`);
        }
    };

    /**
     * Handle game over
     */
    const handleGameOver = async () => {
        const state = getState();

        try {
            const result = await API.getGameResult(state.gameId);
            UI.displayGameResult(_normalizeResultDisplayNames(result));
        } catch (error) {
            console.error('Failed to get result:', error);
            UI.displayGameResult({
                winner: 'Errore',
                points: {}
            });
        }

        API.disconnectWebSocket();
        _stopPolling();  // a partita finita ferma anche il polling (default cloud): niente GET inutili
        store.setState({ connected: false });
        UI.updateGameInfo({ connected: false });
    };

    /**
     * Reset and start over
     */
    const resetGame = () => {
        API.disconnectWebSocket();
        _stopPolling();

        if (initialBoardReadyTimeoutId) {
            clearTimeout(initialBoardReadyTimeoutId);
            initialBoardReadyTimeoutId = null;
        }
        initialBoardReadyResolve = null;

        store.setState({
            gameId: null,
            playerName: null,
            playerIndex: 0,
            opponentIndex: 1,
            connected: false,
            observation: null,
            gameOver: false
        });

        lastAppliedServerVersion = -1;
        pendingEvents = [];
        uiHoldUntilMs = 0;
        startupOverlayReleaseAtMs = 0;
        startupEventsHoldUntilMs = 0;
        debugPeekActive = false;
        debugPeekRequestId += 1;
        activeOpponentReveal = null;
        startingPlayerAnnounced = false;
        my_turn_started_at_ms = null;
        my_turn_observation_server_version = null;
        was_my_turn = false;
        UI.showGameSetup();
    };

    const abandonGame = async () => {
        const state = getState();
        if (!state.gameId || state.gameOver) return;

        const gameId = state.gameId;
        const playerIndex = state.playerIndex;
        try {
            await API.abandonGame(gameId, playerIndex);
        } catch (error) {
            console.warn('Abbandono partita non confermato dal server:', error?.message || error);
        } finally {
            resetGame();
        }
    };

    // Initialize
    UI.init({
        onPrepareStart: loadAiCatalogs,
        onStartGame: startGame,
        onAdvancedOptionsOpen: loadAiCatalogs,
        onAbandonGame: abandonGame,
        onNewGame: resetGame
    });

    // Avviso cold start cloud: quando una richiesta REST resta appesa oltre la soglia
    // (replica che si sveglia dallo scale-to-zero o scale-up in corso), la UI mostra
    // un messaggio di cortesia non bloccante; il layer API lo spegne alla risposta.
    // Registrato PRIMA dei fetch di metadati, così copre anche il primissimo giro.
    API.setSlowRequestListener((active) => UI.setServerWakeNotice(active));

    loadRuntimeMetadata();

    // Precarica le immagini delle carte in background mentre l'utente è sulla home:
    // quando avvia la partita sono già in cache (niente flicker al primo render).
    UI.preloadCardAssets();

    document.addEventListener('keydown', (event) => {
        if (event.repeat || _isTypingTarget(event.target)) return;
        if ((event.key || '').toLowerCase() === 's') {
            _setDebugPeekActive(true);
        }
    });

    document.addEventListener('keyup', (event) => {
        if ((event.key || '').toLowerCase() === 's') {
            _setDebugPeekActive(false);
        }
    });

    UI.showGameSetup();
});
