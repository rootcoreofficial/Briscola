/**
 * Modulo UI per Briscola AI - Versione Semplificata
 *
 * Gestisce il rendering della UI. Non contiene logica di gioco.
 */

const UI = (() => {
    const CARD_ASSET_BASE = '/static/assets/cards';
    const DATA_CONSENT_STORAGE_KEY = 'briscola_data_collection_consent';
    const PLAYER_NAME_STORAGE_KEY = 'briscola_player_name';
    const AI_AGENT_STORAGE_KEY = 'briscola_last_ai_agent';
    const AI_MODEL_STORAGE_KEY = 'briscola_last_ai_model';
    const ADVANCED_OPTIONS_STORAGE_KEY = 'briscola_advanced_options_open';
    const DEFAULT_PLAYER_NAME = 'Giocatore';
    const DEFAULT_AI_AGENT_NAME = 'bc_model_pimc_belief_12x8';

    // Avviso non bloccante per il cold start del server cloud (scale-to-zero).
    // Il testo del dettaglio overlay deve combaciare con quello in index.html.
    const SERVER_WAKE_MESSAGE = 'Il server si sta svegliando, un attimo…';
    const STARTUP_LOADING_DETAIL_DEFAULT = 'Caricamento tavolo e IA';
    const STARTUP_TIPS = [
        'Non regalare Assi e Tre se l’avversario può ancora tagliare.',
        'Una briscola piccola può valere più di una carta alta se ti fa prendere il momento giusto.',
        'Se il piatto vale poco, non sprecare una briscola grossa.',
        'Quando sei secondo di mano, guarda prima quanto vale la presa: non tutto merita di essere vinto.',
        'Tenere una briscola per il finale spesso conta più che vincere una presa subito.',
        'Se guidi un carico, chiediti sempre: “e se l’altro taglia?”',
        'Le carte lisce non sono inutili: a volte servono proprio per non scoprire il gioco.',
        'Quando il mazzo sta finendo, ogni briscola rimasta pesa di più.',
        'Non inseguire tutti i punti: alcune prese si lasciano andare per vincerne una migliore dopo.',
        'La Briscola non premia solo chi prende tanto, ma chi prende al momento giusto.',
    ];

    // Map rank names to numbers for image paths
    const RANK_TO_NUMBER = {
        ACE: 1, TWO: 2, THREE: 3, FOUR: 4, FIVE: 5,
        SIX: 6, SEVEN: 7, JACK: 8, KNIGHT: 9, KING: 10
    };

    // DOM elements cache
    const elements = {
        homeHero: document.getElementById('home-hero'),
        homeAbout: document.getElementById('home-about'),
        startupLoading: document.getElementById('startup-loading'),
        startupLoadingDetail: document.getElementById('startup-loading-detail'),
        startupTipText: document.getElementById('startup-tip-text'),
        gameSetup: document.getElementById('game-setup'),
        gameBoard: document.getElementById('game-board'),
        gameResult: document.getElementById('game-result'),
        gameForm: document.getElementById('game-form'),
        startGameButton: document.getElementById('start-game'),
        setupError: document.getElementById('setup-error'),
        playerNameInput: document.getElementById('player-name-input'),
        defaultOpponentSummary: document.getElementById('default-opponent-summary'),
        advancedOptionsToggle: document.getElementById('advanced-options-toggle'),
        advancedOptions: document.getElementById('advanced-options'),
        advancedOptionsLoading: document.getElementById('advanced-options-loading'),
        aiAgentSelect: document.getElementById('ai-agent-select'),
        aiAgentDescription: document.getElementById('ai-agent-description'),
        aiAgentCommonNote: document.getElementById('ai-agent-common-note'),
        aiAgentCommonNoteText: document.getElementById('ai-agent-common-note-text'),
        aiModelGroup: document.getElementById('ai-model-group'),
        aiModelSelect: document.getElementById('ai-model-select'),
        aiModelDescription: document.getElementById('ai-model-description'),
        dataConsentGroup: document.getElementById('data-consent-group'),
        dataConsentCheckbox: document.getElementById('data-consent-checkbox'),
        dataConsentDescription: document.getElementById('data-consent-description'),
        gameId: document.getElementById('game-id'),
        gameStatus: document.getElementById('game-status'),
        opponentName: document.getElementById('opponent-name'),
        opponentPoints: document.getElementById('opponent-points'),
        opponentHand: document.getElementById('opponent-hand'),
        playerNameDisplay: document.getElementById('player-name-display'),
        playerPoints: document.getElementById('player-points'),
        playerHand: document.getElementById('player-hand'),
        turnIndicator: document.getElementById('turn-indicator'),
        deck: document.getElementById('deck'),
        deckCount: document.getElementById('deck-count'),
        trumpCard: document.getElementById('trump-card'),
        tableCards: document.getElementById('table-cards'),
        turnMessage: document.getElementById('turn-message'),
        trickResult: document.getElementById('trick-result'),
        resultContent: document.getElementById('result-content'),
        newGame: document.getElementById('new-game'),
        abandonGame: document.getElementById('abandon-game'),
        abandonModal: document.getElementById('abandon-modal'),
        abandonCancel: document.getElementById('abandon-cancel'),
        abandonConfirm: document.getElementById('abandon-confirm')
    };

    // Metadati agenti IA caricati dal backend (source of truth: modulo Python).
    let aiAgentMetaByName = {};

    // Modelli locali selezionabili dagli agenti che richiedono un `.npz`.
    let aiModelMetaById = {};
    let recommendedAiModelId = '';

    // Se true, richiediamo una checkbox esplicita prima di avviare la partita.
    let dataConsentRequired = false;
    let gameStartupInProgress = false;
    let advancedOptionsOpen = false;
    let abandonModalOpen = false;
    let uiCallbacks = {};

    // Avviso "server che si sveglia": quando è attivo, il badge di stato nell'header
    // mostra il messaggio di cortesia AL POSTO dello stato connessione corrente.
    // Ricordiamo l'ultimo stato "base" (quello richiesto da game.js) per poterlo
    // ripristinare fedelmente quando l'avviso si spegne, anche se nel frattempo lo
    // stato è cambiato (es. una riconnessione WS conclusa mentre l'avviso era su).
    let serverWakeNoticeActive = false;
    let lastGameStatusRender = { text: 'Non connesso', className: '' };

    const _readStoredDataConsent = () => {
        try {
            return window.localStorage.getItem(DATA_CONSENT_STORAGE_KEY) === 'true';
        } catch (e) {
            // Privacy mode o storage disabilitato: la checkbox resta una scelta per-sessione.
            return false;
        }
    };

    const _writeStoredDataConsent = (accepted) => {
        try {
            if (accepted) {
                window.localStorage.setItem(DATA_CONSENT_STORAGE_KEY, 'true');
            } else {
                window.localStorage.removeItem(DATA_CONSENT_STORAGE_KEY);
            }
        } catch (e) {
            // Il consenso viene comunque inviato nel payload della partita corrente.
        }
    };

    const _readStoredSelection = (key) => {
        try {
            const value = window.localStorage.getItem(key);
            return typeof value === 'string' && value.length > 0 ? value : '';
        } catch (e) {
            // Privacy mode o storage disabilitato: si usa il default.
            return '';
        }
    };

    const _readStoredAdvancedOptions = () => {
        try {
            return window.localStorage.getItem(ADVANCED_OPTIONS_STORAGE_KEY) === 'true';
        } catch (e) {
            return false;
        }
    };

    const _writeStoredAdvancedOptions = (open) => {
        try {
            window.localStorage.setItem(ADVANCED_OPTIONS_STORAGE_KEY, open ? 'true' : 'false');
        } catch (e) {
            // Storage disabilitato: il toggle resta valido per la sessione corrente.
        }
    };

    const _writeStoredSelection = (key, value) => {
        try {
            if (value) {
                window.localStorage.setItem(key, value);
            } else {
                window.localStorage.removeItem(key);
            }
        } catch (e) {
            // Storage disabilitato: la selezione resta valida per la sessione corrente.
        }
    };

    const _readStoredPlayerName = () => {
        try {
            const value = window.localStorage.getItem(PLAYER_NAME_STORAGE_KEY);
            return typeof value === 'string' && value.trim().length > 0 ? value.trim() : '';
        } catch (e) {
            return '';
        }
    };

    const _writeStoredPlayerName = (name) => {
        try {
            const value = typeof name === 'string' ? name.trim() : '';
            if (value) {
                window.localStorage.setItem(PLAYER_NAME_STORAGE_KEY, value);
            } else {
                window.localStorage.removeItem(PLAYER_NAME_STORAGE_KEY);
            }
        } catch (e) {
            // Storage disabilitato: il nome resta valido per la partita corrente.
        }
    };

    const _restorePlayerNameInput = () => {
        if (!elements.playerNameInput) return;
        const storedName = _readStoredPlayerName();
        if (storedName) elements.playerNameInput.value = storedName;
    };

    const _currentPlayerName = () => {
        const raw = elements.playerNameInput?.value || '';
        return raw.trim() || DEFAULT_PLAYER_NAME;
    };

    const _handlePlayerNameChange = () => {
        _writeStoredPlayerName(elements.playerNameInput?.value || '');
    };

    const _restoreDataConsentCheckbox = () => {
        if (!elements.dataConsentCheckbox) return;
        elements.dataConsentCheckbox.checked = _readStoredDataConsent();
    };

    const _handleDataConsentChange = () => {
        _writeStoredDataConsent(elements.dataConsentCheckbox?.checked === true);
        _updateConsentUi();
    };

    const _selectedOptionText = (selectEl, fallback) => {
        const text = selectEl?.selectedOptions?.[0]?.textContent || fallback;
        return text.replace(/\s+/g, ' ').trim();
    };

    const _updateDefaultOpponentSummary = () => {
        if (!elements.defaultOpponentSummary) return;
        const agentName = elements.aiAgentSelect?.value || DEFAULT_AI_AGENT_NAME;
        const label = _selectedOptionText(elements.aiAgentSelect, 'IA consigliata');
        const isDefault = agentName === DEFAULT_AI_AGENT_NAME;
        elements.defaultOpponentSummary.textContent = isDefault
            ? 'Avversario: IA consigliata, il modello più aggiornato del progetto. Puoi cambiarlo nelle opzioni avanzate.'
            : `Avversario: ${label}. Puoi cambiarlo nelle opzioni avanzate.`;
    };

    const _updateAiAgentDescription = () => {
        const name = elements.aiAgentSelect?.value;
        const meta = name ? aiAgentMetaByName[name] : null;
        if (elements.aiAgentDescription) elements.aiAgentDescription.textContent = meta?.description_it || '';
        _updateDefaultOpponentSummary();
    };

    const _isBestAiModel = (model) => {
        const id = model?.id || '';
        const filename = model?.filename || '';
        return Boolean(recommendedAiModelId) && (id === recommendedAiModelId || filename === recommendedAiModelId);
    };

    const _agentRequiresModelSelection = (agentName) => {
        const meta = agentName ? aiAgentMetaByName[agentName] : null;
        return (
            meta?.requires_model_selection === true ||
            agentName === 'bc_model' ||
            agentName === 'bc_model_hybrid_endgame' ||
            agentName === 'bc_model_value_lookahead_8x8'
        );
    };

    const _modelRecencyScore = (model) => {
        const id = model?.id || model?.filename || '';
        const match = id.match(/best_a2c_v(\d+)\.npz$/);
        if (match) return Number.parseInt(match[1], 10);
        if (id === 'best_a2c.npz') return 2;
        return 0;
    };

    const _formatAiModelOptionLabel = (model) => {
        const label = model?.label || model?.filename || model?.id || 'Modello locale';
        const filename = model?.filename || model?.id || '';
        const suffix = filename && filename !== label ? ` (${filename})` : '';
        const prefix = _isBestAiModel(model) ? 'Consigliato - ' : '';
        return `${prefix}${label}${suffix}`;
    };

    const _formatAiModelDescription = (model) => {
        if (!model) return '';

        const lines = [];
        if (_isBestAiModel(model)) {
            lines.push('Stato: best attuale consigliato');
        }

        const filename = model.filename || model.id;
        if (filename) {
            lines.push(`File: ${filename}`);
        }

        const guard = model.metadata?.inference_overkill_guard ?? model.metadata?.inference?.overkill_guard;
        if (typeof guard === 'boolean') {
            lines.push(`Guard anti-overkill: ${guard ? 'attivo' : 'non attivo'}`);
        }

        const desc = model.description_it || '';
        if (desc) {
            lines.push(desc);
        }
        return lines.join('\n');
    };

    const _updateConsentUi = () => {
        if (!elements.startGameButton) return;
        if (gameStartupInProgress) {
            elements.startGameButton.disabled = true;
            return;
        }
        const checked = elements.dataConsentCheckbox?.checked === true;
        elements.startGameButton.disabled = !checked;
    };

    const _updateAiModelUi = () => {
        const agentName = elements.aiAgentSelect?.value;
        const requiresModel = _agentRequiresModelSelection(agentName);

        if (elements.aiModelGroup) {
            elements.aiModelGroup.classList.toggle('hidden', !requiresModel);
        }

        if (!requiresModel) {
            if (elements.aiModelDescription) elements.aiModelDescription.textContent = '';
            return;
        }

        const modelId = elements.aiModelSelect?.value;
        const meta = modelId ? aiModelMetaById[modelId] : null;
        if (elements.aiModelDescription) {
            const desc = _formatAiModelDescription(meta);
            const compatible = meta?.is_compatible;
            const reason = meta?.compatibility_reason_it || '';
            if (compatible === false) {
                elements.aiModelDescription.textContent = `${desc}\n\nNON compatibile: ${reason}`.trim();
            } else {
                elements.aiModelDescription.textContent = desc;
            }
        }
        _updateDefaultOpponentSummary();
    };

    const _updateStartupTip = () => {
        if (!elements.startupTipText || STARTUP_TIPS.length === 0) return;
        const index = Math.floor(Math.random() * STARTUP_TIPS.length);
        elements.startupTipText.textContent = STARTUP_TIPS[index];
    };

    const _setGameStartupLoading = (isLoading) => {
        gameStartupInProgress = isLoading === true;
        if (gameStartupInProgress) {
            _updateStartupTip();
        }
        if (elements.startupLoading) {
            elements.startupLoading.classList.toggle('hidden', !gameStartupInProgress);
            elements.startupLoading.setAttribute('aria-hidden', String(!gameStartupInProgress));
        }
        document.body.classList.toggle('starting-game', gameStartupInProgress);
        if (elements.startGameButton) {
            elements.startGameButton.setAttribute('aria-busy', String(gameStartupInProgress));
        }
        _updateConsentUi();
    };

    const _setAdvancedOptionsOpen = (open, { persist = true } = {}) => {
        advancedOptionsOpen = open === true;
        elements.advancedOptions?.classList.toggle('hidden', !advancedOptionsOpen);
        if (elements.advancedOptionsToggle) {
            elements.advancedOptionsToggle.setAttribute('aria-expanded', String(advancedOptionsOpen));
            elements.advancedOptionsToggle.textContent = advancedOptionsOpen ? 'Nascondi opzioni' : 'Opzioni avanzate';
        }
        if (persist) _writeStoredAdvancedOptions(advancedOptionsOpen);
        if (advancedOptionsOpen) {
            Promise.resolve(uiCallbacks.onAdvancedOptionsOpen?.()).catch((error) => {
                showSetupError(error?.message || 'Impossibile caricare le opzioni avanzate.');
            });
        }
    };

    const _clearSetupError = () => {
        if (!elements.setupError) return;
        elements.setupError.textContent = '';
        elements.setupError.classList.add('hidden');
    };

    const _setAbandonModalOpen = (open) => {
        const wasOpen = abandonModalOpen;
        abandonModalOpen = open === true;
        if (elements.abandonModal) {
            elements.abandonModal.classList.toggle('hidden', !abandonModalOpen);
            elements.abandonModal.setAttribute('aria-hidden', String(!abandonModalOpen));
        }
        document.body.classList.toggle('modal-open', abandonModalOpen);
        if (abandonModalOpen) {
            elements.abandonCancel?.focus();
        } else if (wasOpen && elements.abandonGame?.offsetParent !== null) {
            elements.abandonGame?.focus();
        }
    };

    const _requestAbandonConfirmation = () => {
        _setAbandonModalOpen(true);
    };

    /**
     * Normalize card data from various backend formats
     */
    const _normalizeCard = (card) => {
        if (!card || typeof card !== 'object') return null;

        const suit = card.suit?.value || card.suit;
        const rankName = card.rank?.name || card.rank;
        const number = card.number || RANK_TO_NUMBER[rankName] || null;

        return { suit, number, points: card.points };
    };

    /**
     * Get card image source
     */
    const _cardImageSrc = (card) => {
        const normalized = _normalizeCard(card);
        if (!normalized?.suit || !normalized?.number) return null;
        return `${CARD_ASSET_BASE}/${normalized.suit}_${normalized.number}.png`;
    };

    /**
     * Precarica TUTTE le immagini delle carte (40 facce + retro) nel browser.
     *
     * Perché: la prima volta che una carta viene mostrata, senza precaricamento il browser deve
     * scaricarla al volo → comparsa "in ritardo"/flicker. Precaricandole (tipicamente mentre l'utente
     * è ancora sulla home) finiscono in cache e il render è immediato.
     *
     * - Memoizzata: parte una volta sola.
     * - Risolve anche in caso di errore di una singola immagine: il preload non deve mai bloccare il gioco.
     */
    let _cardPreloadPromise = null;
    const preloadCardAssets = () => {
        if (_cardPreloadPromise) return _cardPreloadPromise;
        const suits = ['clubs', 'coins', 'cups', 'swords'];
        const urls = [`${CARD_ASSET_BASE}/card_back.png`];
        for (const suit of suits) {
            for (let n = 1; n <= 10; n++) urls.push(`${CARD_ASSET_BASE}/${suit}_${n}.png`);
        }
        _cardPreloadPromise = Promise.all(
            urls.map(
                (url) =>
                    new Promise((resolve) => {
                        const img = new Image();
                        img.onload = () => resolve();
                        img.onerror = () => resolve();
                        img.src = url;
                    })
            )
        );
        return _cardPreloadPromise;
    };

    // Nomi italiani per le etichette accessibili delle carte (alt/aria-label).
    const SUIT_NAMES_IT = { clubs: 'Bastoni', cups: 'Coppe', coins: 'Denari', swords: 'Spade' };
    const RANK_NAMES_IT = {
        ACE: 'Asso', TWO: 'Due', THREE: 'Tre', FOUR: 'Quattro', FIVE: 'Cinque',
        SIX: 'Sei', SEVEN: 'Sette', JACK: 'Fante', KNIGHT: 'Cavallo', KING: 'Re'
    };

    const cardLabelIt = (card) => {
        if (!card) return 'Carta coperta';
        const rank = RANK_NAMES_IT[card.rank] || card.rank;
        const suit = SUIT_NAMES_IT[card.suit] || card.suit;
        return `${rank} di ${suit}`;
    };

    /**
     * Create a card element.
     *
     * Accessibilità: le carte giocabili sono veri controlli (role="button", tabbabili,
     * attivabili con Invio/Spazio), non solo div cliccabili col mouse.
     */
    const createCardElement = (card, onClick = null) => {
        const cardEl = document.createElement('div');
        cardEl.className = 'card';

        if (!card) {
            // Face down card
            cardEl.classList.add('card-back');
            cardEl.setAttribute('aria-label', 'Carta coperta');
            return cardEl;
        }

        const label = cardLabelIt(card);
        const src = _cardImageSrc(card);
        if (src) {
            const img = document.createElement('img');
            img.className = 'card-face';
            img.src = src;
            img.alt = label;
            img.loading = 'lazy';
            cardEl.appendChild(img);
        } else {
            cardEl.classList.add('card-back');
        }

        if (onClick) {
            cardEl.classList.add('clickable');
            cardEl.setAttribute('role', 'button');
            cardEl.setAttribute('tabindex', '0');
            cardEl.setAttribute('aria-label', `Gioca ${label}`);
            cardEl.addEventListener('click', onClick);
            cardEl.addEventListener('keydown', (event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    onClick(event);
                }
            });
        } else {
            cardEl.classList.add('disabled');
            cardEl.setAttribute('aria-label', label);
        }

        return cardEl;
    };

    // --- Public API ---

    const init = (callbacks) => {
        uiCallbacks = callbacks || {};
        _restorePlayerNameInput();
        _setAdvancedOptionsOpen(_readStoredAdvancedOptions(), { persist: false });
        _updateDefaultOpponentSummary();

        elements.gameForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (gameStartupInProgress || !callbacks.onStartGame) return;
            _clearSetupError();
            _setGameStartupLoading(true);
            try {
                await callbacks.onPrepareStart?.();
                const modelId = elements.aiModelSelect?.value || null;
                const modelMeta = modelId ? aiModelMetaById[modelId] : null;
                const playerName = _currentPlayerName();
                const aiAgent = elements.aiAgentSelect?.value || DEFAULT_AI_AGENT_NAME;
                _writeStoredPlayerName(playerName);
                await callbacks.onStartGame({
                    playerName,
                    aiAgent,
                    aiAgentLabel: _selectedOptionText(elements.aiAgentSelect, 'IA consigliata'),
                    aiAgentRequiresModelSelection: _agentRequiresModelSelection(aiAgent),
                    aiModelId: modelId,
                    aiModelLabel: _selectedOptionText(elements.aiModelSelect, '') || null,
                    aiModelCompatible: modelMeta?.is_compatible === true,
                    aiModelCompatibilityReasonIt: modelMeta?.compatibility_reason_it || null,
                    consentToDataCollection: elements.dataConsentCheckbox?.checked === true,
                });
            } finally {
                _setGameStartupLoading(false);
            }
        });

        elements.aiAgentSelect?.addEventListener('change', () => {
            _writeStoredSelection(AI_AGENT_STORAGE_KEY, elements.aiAgentSelect?.value || '');
            _updateAiAgentDescription();
            _updateAiModelUi();
        });
        elements.aiModelSelect?.addEventListener('change', () => {
            _writeStoredSelection(AI_MODEL_STORAGE_KEY, elements.aiModelSelect?.value || '');
            _updateAiModelUi();
        });
        elements.playerNameInput?.addEventListener('change', _handlePlayerNameChange);
        elements.dataConsentCheckbox?.addEventListener('change', _handleDataConsentChange);
        elements.advancedOptionsToggle?.addEventListener('click', () => {
            _setAdvancedOptionsOpen(!advancedOptionsOpen);
        });

        elements.abandonGame?.addEventListener('click', () => {
            _requestAbandonConfirmation();
        });
        elements.abandonCancel?.addEventListener('click', () => {
            _setAbandonModalOpen(false);
        });
        elements.abandonConfirm?.addEventListener('click', () => {
            _setAbandonModalOpen(false);
            callbacks.onAbandonGame?.();
        });
        elements.abandonModal?.addEventListener('click', (event) => {
            if (event.target === elements.abandonModal) {
                _setAbandonModalOpen(false);
            }
        });
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && abandonModalOpen) {
                event.preventDefault();
                _setAbandonModalOpen(false);
            }
        });

        elements.newGame?.addEventListener('click', () => {
            callbacks.onNewGame?.();
        });
    };

    const setAiAgents = (catalog) => {
        const agents = Array.isArray(catalog) ? catalog : (catalog?.agents || []);

        if (!elements.aiAgentSelect || !Array.isArray(agents) || agents.length === 0) {
            _updateAiAgentDescription();
            _updateAiModelUi();
            return;
        }

        aiAgentMetaByName = {};
        agents.forEach((a) => {
            if (a?.name) aiAgentMetaByName[a.name] = a;
        });

        elements.aiAgentSelect.innerHTML = '';
        agents.forEach((a) => {
            if (!a?.name) return;
            const option = document.createElement('option');
            option.value = a.name;
            const available = a.available !== false;
            // Opzioni non disponibili (es. modello richiesto assente nel deploy) restano visibili
            // ma disabilitate: l'utente capisce che esistono ma non può selezionarle (niente errori).
            option.textContent = available ? (a.label || a.name) : `${a.label || a.name} (non disponibile)`;
            if (!available) {
                option.disabled = true;
                option.title = 'Modello richiesto non disponibile in questo deploy';
            }
            elements.aiAgentSelect.appendChild(option);
        });

        // Default: l'ultima scelta dell'utente (localStorage) se ancora disponibile;
        // altrimenti il PIMC belief 12x8 promosso con v15, con fallback a scendere.
        const isAvail = (name) => !!(name && aiAgentMetaByName[name] && aiAgentMetaByName[name].available !== false);
        const firstAvailable = agents.find((a) => a?.name && a.available !== false)?.name;
        const stored = advancedOptionsOpen ? _readStoredSelection(AI_AGENT_STORAGE_KEY) : '';
        let defaultAgent;
        if (isAvail(stored)) defaultAgent = stored;
        else if (isAvail(DEFAULT_AI_AGENT_NAME)) defaultAgent = DEFAULT_AI_AGENT_NAME;
        else if (isAvail('bc_model_pimc_belief_64x10')) defaultAgent = 'bc_model_pimc_belief_64x10';
        else if (isAvail('bc_model')) defaultAgent = 'bc_model';
        else if (isAvail('heuristic_v1')) defaultAgent = 'heuristic_v1';
        else defaultAgent = firstAvailable || agents[0]?.name || 'random';
        elements.aiAgentSelect.value = defaultAgent;
        _updateAiAgentDescription();
        _updateAiModelUi();
    };

    /**
     * Imposta la lista di modelli `.npz` disponibili (per l'agente `bc_model`).
     *
     * Payload atteso:
     * - `[{ id, label, description_it, ... }]`
     * - oppure `{ models: [...] }`
     */
    const setAiModels = (catalog) => {
        const models = Array.isArray(catalog) ? catalog : (catalog?.models || []);
        recommendedAiModelId = Array.isArray(catalog) ? '' : (catalog?.recommended_model || '');
        aiModelMetaById = {};

        if (!elements.aiModelSelect) {
            _updateAiModelUi();
            return;
        }

        elements.aiModelSelect.innerHTML = '';
        if (!Array.isArray(models) || models.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'Nessun modello trovato';
            option.disabled = true;
            option.selected = true;
            elements.aiModelSelect.appendChild(option);
            _updateAiModelUi();
            return;
        }

        const orderedModels = [...models].sort((a, b) => {
            if (_isBestAiModel(a) && !_isBestAiModel(b)) return -1;
            if (!_isBestAiModel(a) && _isBestAiModel(b)) return 1;
            const scoreDiff = _modelRecencyScore(b) - _modelRecencyScore(a);
            if (scoreDiff !== 0) return scoreDiff;
            return (a?.label || a?.id || '').localeCompare(b?.label || b?.id || '');
        });

        orderedModels.forEach((m) => {
            if (!m?.id) return;
            aiModelMetaById[m.id] = m;
            const option = document.createElement('option');
            option.value = m.id;
            option.textContent = _formatAiModelOptionLabel(m);
            if (m.is_compatible === false) {
                option.disabled = true;
                const reason = m.compatibility_reason_it ? ` (${m.compatibility_reason_it})` : '';
                option.textContent = `NON COMPATIBILE: ${option.textContent}${reason}`;
            }
            elements.aiModelSelect.appendChild(option);
        });

        // Default: ultima scelta dell'utente (localStorage) se ancora presente e compatibile;
        // altrimenti modello consigliato compatibile; fallback al best più recente compatibile.
        const storedModel = advancedOptionsOpen ? _readStoredSelection(AI_MODEL_STORAGE_KEY) : '';
        const storedCompatible = orderedModels.find((m) => m?.id === storedModel && m.is_compatible !== false);
        const recommendedCompatible = orderedModels.find((m) => _isBestAiModel(m) && m?.id && m.is_compatible !== false);
        const firstCompatible = orderedModels.find((m) => m?.id && m.is_compatible !== false);
        if (storedCompatible?.id) elements.aiModelSelect.value = storedCompatible.id;
        else if (recommendedCompatible?.id) elements.aiModelSelect.value = recommendedCompatible.id;
        else if (firstCompatible?.id) elements.aiModelSelect.value = firstCompatible.id;
        else if (orderedModels[0]?.id) elements.aiModelSelect.value = orderedModels[0].id;

        _updateAiModelUi();
        _updateDefaultOpponentSummary();
    };

    const setDataCollectionConsent = (payload) => {
        const required = payload?.required === true;
        const descriptionIt = payload?.description_it || '';
        dataConsentRequired = true;

        if (elements.dataConsentGroup) {
            elements.dataConsentGroup.classList.remove('hidden');
        }
        _restoreDataConsentCheckbox();
        if (elements.dataConsentDescription) {
            elements.dataConsentDescription.textContent =
                descriptionIt ||
                'Serve il consenso: le tue mosse vengono registrate in forma anonima per migliorare e valutare l’IA.';
        }
        _updateConsentUi();
    };

    const setAdvancedOptionsLoading = (loading) => {
        elements.advancedOptionsLoading?.classList.toggle('hidden', loading !== true);
    };

    const showSetupError = (message) => {
        if (!elements.setupError) return;
        const text = String(message || '').trim();
        elements.setupError.textContent = text;
        elements.setupError.classList.toggle('hidden', text.length === 0);
    };

    const showGameSetup = () => {
        _setGameStartupLoading(false);
        _setAbandonModalOpen(false);
        _clearSetupError();
        elements.homeHero?.classList.remove('hidden');
        elements.homeAbout?.classList.remove('hidden');
        elements.gameSetup.classList.remove('hidden');
        elements.gameBoard.classList.add('hidden');
        elements.gameResult.classList.add('hidden');
        document.body.classList.remove('playing');
        // Il consenso resta revocabile: se l'utente lo ha già dato, riproponiamo la checkbox selezionata.
        _restoreDataConsentCheckbox();
        _updateConsentUi();
    };

    const showGameBoard = () => {
        elements.homeHero?.classList.add('hidden');
        elements.homeAbout?.classList.add('hidden');
        elements.gameSetup.classList.add('hidden');
        elements.gameBoard.classList.remove('hidden');
        elements.gameResult.classList.add('hidden');
        // `playing` abilita il layout "fit-to-viewport" su mobile (vedi CSS): solo in partita,
        // così setup e risultato restano scrollabili normalmente.
        document.body.classList.add('playing');
    };

    const showGameResult = () => {
        _setAbandonModalOpen(false);
        elements.homeHero?.classList.add('hidden');
        elements.homeAbout?.classList.add('hidden');
        elements.gameSetup.classList.add('hidden');
        elements.gameBoard.classList.add('hidden');
        elements.gameResult.classList.remove('hidden');
        document.body.classList.remove('playing');
    };

    /**
     * Aggiorna informazioni "header" della partita (id + stato connessione).
     *
     * Nota:
     * - `connected` è utile come boolean base.
     * - `statusText`/`statusClass` permettono uno stato più granulare (es. "Riconnessione...").
     */
    const _renderGameStatus = (text, className) => {
        elements.gameStatus.textContent = text;
        elements.gameStatus.className = className;
    };

    const updateGameInfo = ({ gameId, connected, statusText, statusClass }) => {
        if (gameId) {
            elements.gameId.textContent = `ID: ${gameId.substring(0, 8)}...`;
        }
        if (connected !== undefined || statusText !== undefined || statusClass !== undefined) {
            const text = statusText !== undefined ? statusText : (connected ? 'Connesso' : 'Non connesso');

            // Manteniamo l'id `game-status` e usiamo classi "stateful" per i colori.
            const classes = [];
            if (statusClass) classes.push(statusClass);
            else if (connected) classes.push('connected');

            // Memorizziamo sempre lo stato base; il render è rinviato se l'avviso
            // "server che si sveglia" sta temporaneamente occupando il badge.
            lastGameStatusRender = { text, className: classes.join(' ') };
            if (!serverWakeNoticeActive) {
                _renderGameStatus(lastGameStatusRender.text, lastGameStatusRender.className);
            }
        }
    };

    /**
     * Mostra/nasconde l'avviso non bloccante "il server si sta svegliando".
     *
     * Chi lo pilota: il layer API (via game.js) quando una richiesta REST supera la
     * soglia di lentezza — tipicamente il cold start del deploy cloud (scale-to-zero).
     *
     * Riusa i canali di stato già esistenti (nessun sistema di toast separato):
     * - il badge `#game-status` nell'header, lo stesso dei messaggi di riconnessione WS;
     * - la riga di dettaglio dell'overlay di avvio partita, perché durante la
     *   creazione della partita (il caso lento più frequente) l'overlay copre l'header.
     */
    const setServerWakeNotice = (active) => {
        const next = active === true;
        if (serverWakeNoticeActive === next) return;
        serverWakeNoticeActive = next;

        if (next) {
            // Stessa classe "warning" usata per i tentativi di riconnessione WS.
            _renderGameStatus(SERVER_WAKE_MESSAGE, 'reconnecting');
        } else {
            _renderGameStatus(lastGameStatusRender.text, lastGameStatusRender.className);
        }

        if (elements.startupLoadingDetail) {
            elements.startupLoadingDetail.textContent = next ? SERVER_WAKE_MESSAGE : STARTUP_LOADING_DETAIL_DEFAULT;
        }
    };

    const renderPlayerHand = (cards, isMyTurn, onCardClick) => {
        elements.playerHand.innerHTML = '';
        cards.forEach((card, index) => {
            const onClick = isMyTurn ? () => onCardClick(index) : null;
            const cardEl = createCardElement(card, onClick);
            cardEl.classList.add('card-appear');
            elements.playerHand.appendChild(cardEl);
        });

        // Show/hide turn indicator (visibility mantiene lo spazio nel layout)
        elements.turnIndicator.style.visibility = isMyTurn ? 'visible' : 'hidden';
    };

    const renderOpponentHand = (cardCount, revealedCards = null) => {
        elements.opponentHand.innerHTML = '';
        const cards = Array.isArray(revealedCards) ? revealedCards : null;
        const count = cards ? cards.length : cardCount;
        for (let i = 0; i < count; i++) {
            const cardEl = createCardElement(cards ? cards[i] : null);
            if (cards) cardEl.classList.add('debug-peek-card');
            elements.opponentHand.appendChild(cardEl);
        }
    };

    /**
     * Reveal a specific card in opponent's hand (show face-up with highlight)
     */
    const revealOpponentCard = (cardIndex, card, decisionType = null) => {
        const cards = elements.opponentHand.children;
        if (cardIndex >= 0 && cardIndex < cards.length) {
            const cardEl = cards[cardIndex];
            // Replace card back with face-up card
            const src = _cardImageSrc(card);
            if (src) {
                cardEl.classList.remove('card-back');
                cardEl.classList.add('revealed');
                cardEl.classList.toggle('revealed-lookahead', decisionType === 'lookahead' || decisionType === 'search');
                cardEl.classList.toggle('revealed-solver', decisionType === 'solver');
                const img = document.createElement('img');
                img.className = 'card-face';
                img.src = src;
                img.alt = 'Carta IA';
                cardEl.innerHTML = '';
                cardEl.appendChild(img);
            }
        } else {
            console.warn('Card index out of range:', cardIndex, 'vs', cards.length);
        }
    };

    /**
     * Evidenzia (lampeggia) la carta scelta dal giocatore nella sua mano.
     *
     * Nota didattica:
     * - il backend è la "single source of truth": questa è solo una micro-animazione
     *   locale per rendere chiaro quale carta è stata selezionata PRIMA che venga
     *   renderizzata sul tavolo tramite l'update WebSocket.
     * - non rimuove la carta dalla mano: la rimozione/aggiornamento arriva dallo snapshot.
     *
     * @param {number} cardIndex - indice della carta nella mano del player
     */
    const revealPlayerCard = (cardIndex) => {
        const cards = elements.playerHand.children;
        if (cardIndex < 0 || cardIndex >= cards.length) return;

        // Metti in evidenza la carta scelta e disabilita visivamente le altre
        // durante l'azione (evita confusione/doppi click).
        Array.from(cards).forEach((cardEl, idx) => {
            cardEl.classList.toggle('revealed', idx === cardIndex);
            cardEl.classList.toggle('disabled', idx !== cardIndex);
        });
    };

    /**
     * Ripristina lo stato visivo della mano del giocatore (rimuove highlight/disabled).
     *
     * Serve quando:
     * - la connessione WS cade durante un'azione
     * - la UI è in "hold" ma lo snapshot successivo non arriva (o arriva in ritardo)
     */
    const resetPlayerHandHighlights = () => {
        elements.playerHand.querySelectorAll('.card').forEach((cardEl) => {
            cardEl.classList.remove('revealed');
            cardEl.classList.remove('disabled');
        });
    };

    /**
     * Remove any revealed card from both player's and opponent's hands.
     * Use this when the card moves to the table.
     */
    const removeRevealedCard = () => {
        // Rimuovi carte evidenziate dalla mano avversario
        elements.opponentHand.querySelectorAll('.revealed').forEach(card => card.remove());
        // Rimuovi carte evidenziate dalla mano del giocatore
        elements.playerHand.querySelectorAll('.revealed').forEach(card => card.remove());
    };

    const renderTableCards = (tableCards) => {
        elements.tableCards.innerHTML = '';

        if (!Array.isArray(tableCards)) return;

        // Nuovo formato DTO: [{card, player_index}, ...]
        tableCards.forEach((item) => {
            const card = item.card;
            const playerIndex = item.player_index;

            const wrapper = document.createElement('div');
            wrapper.className = 'table-card';

            const cardEl = createCardElement(card);
            cardEl.classList.add('card-appear');
            wrapper.appendChild(cardEl);

            // Label
            const label = document.createElement('div');
            label.className = 'card-label';
            label.textContent = playerIndex === 0 ? 'Tu' : 'IA';
            wrapper.appendChild(label);

            elements.tableCards.appendChild(wrapper);
        });
    };

    const renderTrumpCard = (card, trumpSuit = null) => {
        elements.trumpCard.innerHTML = '';
        if (card) {
            const cardEl = createCardElement(card);
            elements.trumpCard.appendChild(cardEl);
            return;
        }

        // Placeholder sempre presente: mantiene stabile il layout anche quando il mazzo si esaurisce.
        // Quando non abbiamo (o non vogliamo mostrare) la carta, mostriamo comunque il seme di briscola (se noto).
        const suitNames = {
            clubs: 'Bastoni',
            cups: 'Coppe',
            coins: 'Denari',
            swords: 'Spade'
        };
        const label = document.createElement('div');
        label.className = 'trump-suit-indicator';
        if (trumpSuit) {
            label.textContent = `Briscola: ${suitNames[trumpSuit] || trumpSuit}`;
        } else {
            label.textContent = 'Briscola';
        }
        elements.trumpCard.appendChild(label);
    };

    const updateDeckCount = (count, debugNextCard = null) => {
        const safeCount = Number.isFinite(count) ? count : 0;
        elements.deckCount.textContent = safeCount;

        // Manteniamo sempre visibile il placeholder del mazzo per evitare che l'area "tavolo"
        // cambi altezza quando il mazzo si esaurisce.
        elements.deck.style.display = 'flex';
        elements.deck.classList.remove('debug-peek-card');

        if (debugNextCard && safeCount > 0) {
            const src = _cardImageSrc(debugNextCard);
            if (src) {
                elements.deck.classList.remove('deck-empty');
                elements.deck.classList.remove('card-back');
                elements.deck.classList.add('debug-peek-card');
                elements.deck.replaceChildren();

                const img = document.createElement('img');
                img.className = 'card-face';
                img.src = src;
                img.alt = 'Prossima carta mazzo';
                elements.deck.appendChild(img);
                elements.deck.appendChild(elements.deckCount);
                return;
            }
        }

        if (elements.deckCount.parentElement !== elements.deck) {
            elements.deck.replaceChildren(elements.deckCount);
        } else {
            Array.from(elements.deck.children).forEach((child) => {
                if (child !== elements.deckCount) child.remove();
            });
        }

        // Quando il mazzo è vuoto:
        // - non vogliamo più mostrare il retro della carta (sembra che ci sia ancora un mazzo)
        // - vogliamo un placeholder "vuoto" simile allo slot briscola.
        elements.deck.classList.toggle('deck-empty', safeCount <= 0);
        elements.deck.classList.toggle('card-back', safeCount > 0);
    };

    const updatePlayerPoints = (points) => {
        elements.playerPoints.textContent = `${points} punti`;
    };

    const updateOpponentInfo = (name, _points) => {
        elements.opponentName.textContent = name;
        // Fairness: NON mostriamo i punti dell'avversario IA. In Briscola il mazzo di prese
        // avversario non è pubblico; mostrarne il totale aiuterebbe l'umano (che dovrebbe contare
        // a mente). Non scriviamo nemmeno il valore nel DOM. Vedi anche `#opponent-points` nel CSS.
    };

    const showTurnMessage = (message, isThinking = false) => {
        elements.turnMessage.textContent = message;
        elements.turnMessage.className = 'turn-message' + (isThinking ? ' thinking' : '');
    };

    const showTrickResult = (message, duration = 2000) => {
        elements.trickResult.textContent = message;
        elements.trickResult.classList.remove('hidden');

        setTimeout(() => {
            elements.trickResult.classList.add('hidden');
        }, duration);
    };

    const displayGameResult = (result) => {
        elements.resultContent.replaceChildren();

        const title = document.createElement('h3');
        title.textContent = result.winner === 'Pareggio'
            ? 'Pareggio!'
            : `${result.winner || 'Risultato'} vince!`;
        elements.resultContent.appendChild(title);

        const scores = document.createElement('div');
        scores.className = 'scores';

        for (const [name, points] of Object.entries(result.points || {})) {
            const item = document.createElement('div');
            item.className = 'score-item';

            const label = document.createElement('div');
            label.className = 'score-label';
            label.textContent = name;

            const value = document.createElement('div');
            value.className = 'score-value';
            value.textContent = String(points);

            item.appendChild(label);
            item.appendChild(value);
            scores.appendChild(item);
        }

        elements.resultContent.appendChild(scores);
        showGameResult();
    };

    const setPlayerName = (name) => {
        elements.playerNameDisplay.textContent = name;
    };

    return {
        init,
        preloadCardAssets,
        setAiAgents,
        setAiModels,
        setDataCollectionConsent,
        setAdvancedOptionsLoading,
        showSetupError,
        showGameSetup,
        showGameBoard,
        showGameResult,
        updateGameInfo,
        setServerWakeNotice,
        renderPlayerHand,
        renderOpponentHand,
        revealOpponentCard,
        revealPlayerCard,
        resetPlayerHandHighlights,
        removeRevealedCard,
        renderTableCards,
        renderTrumpCard,
        updateDeckCount,
        updatePlayerPoints,
        updateOpponentInfo,
        showTurnMessage,
        showTrickResult,
        displayGameResult,
        setPlayerName
    };
})();
