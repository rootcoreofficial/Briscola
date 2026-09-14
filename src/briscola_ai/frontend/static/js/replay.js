/**
 * Assistente touch per una Briscola fisica, senza pulsanti. Lo stato vive interamente
 * nel browser; il server riceve soltanto un'istantanea anonima quando deve valutare
 * una decisione già registrata.
 */
(() => {
    'use strict';

    const surface = document.getElementById('gesture-surface');
    const statusNode = document.getElementById('replay-status');
    const detailNode = document.getElementById('replay-detail');
    const cardNode = document.getElementById('replay-card');
    const enteredCardSection = document.getElementById('entered-card-section');
    const adviceCardSection = document.getElementById('advice-card-section');
    const adviceCardNode = document.getElementById('advice-card');
    const confirmationNode = document.getElementById('replay-confirmation');

    const SUIT_BY_DIRECTION = {
        up: 'coins',
        left: 'cups',
        right: 'swords',
        down: 'clubs',
    };
    const SUIT_LABEL = { clubs: 'Bastoni', cups: 'Coppe', coins: 'Denari', swords: 'Spade' };
    const RANK_LABEL = { 1: 'Asso', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: 'Fante', 9: 'Cavallo', 10: 'Re' };
    const STRENGTH = { 1: 10, 3: 9, 10: 8, 9: 7, 8: 6, 7: 5, 6: 4, 5: 3, 4: 2, 2: 1 };
    const POINTS = { 1: 11, 3: 10, 10: 4, 9: 3, 8: 2, 2: 0, 4: 0, 5: 0, 6: 0, 7: 0 };
    const TAP_IDLE_MS = 800;
    const OPENING_IDLE_MS = 650;
    const SWIPE_DISTANCE_PX = 48;
    // Nell'APK la pagina e' locale (`capacitor://`), mentre il modello resta sul
    // backend pubblico. Nel browser normale il path relativo evita CORS e conserva
    // la possibilita' di eseguire tutto in locale con `briscola-server`.
    const REPLAY_ADVICE_URL = window.Capacitor?.isNativePlatform?.()
        ? 'https://briscola-m2lw.onrender.com/api/replay/advice'
        : '/api/replay/advice';

    /** Tutta la partita è una registrazione locale: niente localStorage e niente persistenza. */
    const replay = {
        phase: 'card',
        target: 'initial',
        initialCards: [],
        hand: [],
        trump: null,
        completed: [],
        table: [],
        deckSize: 34,
        opponentHandSize: 3,
        points: [0, 0],
        firstPlayer: 0,
        selectedSuit: null,
        taps: 0,
        tapTimer: null,
        recommendation: null,
        pendingCard: null,
    };
    let pointerStart = null;
    let openingTaps = 0;
    let openingTimer = null;
    let audioContext = null;

    const cardKey = (card) => `${card.suit}:${card.number}`;
    const sameCard = (a, b) => a && b && cardKey(a) === cardKey(b);
    const cardLabel = (card) => `${RANK_LABEL[card.number]} di ${SUIT_LABEL[card.suit]}`;

    /**
     * Renderizza solo le informazioni utili al gesto che l'utente deve fare ora.
     * La schermata non conserva indicatori di punteggio o stato: durante una
     * partita fisica ruberebbero spazio alla carta appena inserita o consigliata.
     */
    function setMessage(status, detail = '', card = '', mode = '') {
        statusNode.textContent = status;
        detailNode.textContent = detail;
        cardNode.textContent = card;
        enteredCardSection.classList.toggle('hidden', !card);
        adviceCardNode.textContent = '';
        adviceCardSection.classList.add('hidden');
        confirmationNode.textContent = '';
        confirmationNode.classList.add('hidden');
        surface.classList.toggle('is-thinking', mode === 'thinking');
        surface.classList.toggle('is-error', mode === 'error');
    }

    /** Mostra il consiglio in una sezione distinta dalla carta appena riconosciuta. */
    function setAdviceMessage(status, detail, advice, confirmation) {
        setMessage(status, detail);
        adviceCardNode.textContent = advice;
        adviceCardSection.classList.toggle('hidden', !advice);
        confirmationNode.textContent = confirmation;
        confirmationNode.classList.toggle('hidden', !confirmation);
    }

    /** Chiede una conferma esplicita prima di registrare una carta appena immessa. */
    function previewCard(card) {
        replay.pendingCard = card;
        replay.phase = 'confirm-card';
        setMessage(
            'Controlla la carta inserita.',
            'Un tocco la conferma. Uno swipe corregge il seme.',
            cardLabel(card),
        );
        confirmationNode.textContent = 'Tocca per confermare';
        confirmationNode.classList.remove('hidden');
    }

    /** Un feedback breve, attivato solo da un gesto dell'utente e quindi compatibile con i browser mobili. */
    function cue(kind) {
        if (navigator.vibrate) navigator.vibrate(kind === 'error' ? [30, 50, 30] : 35);
        try {
            audioContext ||= new AudioContext();
            const oscillator = audioContext.createOscillator();
            const gain = audioContext.createGain();
            oscillator.frequency.value = kind === 'error' ? 190 : 740;
            gain.gain.setValueAtTime(0.035, audioContext.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + 0.09);
            oscillator.connect(gain).connect(audioContext.destination);
            oscillator.start();
            oscillator.stop(audioContext.currentTime + 0.1);
        } catch (_error) {
            // Audio è un extra: l'app resta utilizzabile anche se un browser lo blocca.
        }
    }

    function resetCardEntry() {
        replay.selectedSuit = null;
        replay.taps = 0;
        clearTimeout(replay.tapTimer);
        replay.tapTimer = null;
    }

    function knownPhysicalCards() {
        return [...replay.hand, ...replay.completed, ...replay.table.map((entry) => entry.card)];
    }

    function reportError(message) {
        cue('error');
        setMessage(message, 'Ripeti lo swipe del seme e poi il valore.', '', 'error');
        resetCardEntry();
    }

    function startCardEntry(target, suit) {
        replay.phase = 'card';
        replay.target = target;
        replay.selectedSuit = suit;
        replay.taps = 0;
        setMessage(`Seme selezionato: ${SUIT_LABEL[suit]}.`, 'Ora tocca il centro 1–6 volte, oppure usa un angolo per 7, Fante, Cavallo o Re.');
    }

    function directionFromDelta(dx, dy) {
        if (Math.abs(dx) > Math.abs(dy)) return dx > 0 ? 'right' : 'left';
        return dy > 0 ? 'down' : 'up';
    }

    function rankFromCorner(x, y) {
        const rect = surface.getBoundingClientRect();
        const horizontal = x / rect.width;
        const vertical = y / rect.height;
        if (horizontal < 0.28 && vertical < 0.28) return 7;
        if (horizontal > 0.72 && vertical < 0.28) return 8;
        if (horizontal < 0.28 && vertical > 0.72) return 9;
        if (horizontal > 0.72 && vertical > 0.72) return 10;
        return null;
    }

    function isCentralTap(x, y) {
        const rect = surface.getBoundingClientRect();
        const horizontal = x / rect.width;
        const vertical = y / rect.height;
        return horizontal >= 0.25 && horizontal <= 0.75 && vertical >= 0.25 && vertical <= 0.75;
    }

    function onCardTap(x, y) {
        if (!replay.selectedSuit) {
            reportError('Prima scegli il seme con uno swipe.');
            return;
        }
        const cornerRank = rankFromCorner(x, y);
        if (cornerRank !== null) {
            previewCard({ suit: replay.selectedSuit, number: cornerRank });
            return;
        }
        if (!isCentralTap(x, y)) {
            reportError('Tocco fuori dalla zona centrale o dagli angoli.');
            return;
        }
        replay.taps += 1;
        if (replay.taps > 6) {
            reportError('Il centro accetta al massimo sei tocchi.');
            return;
        }
        setMessage(`Valore in corso: ${replay.taps}.`, 'Attendi un istante per confermare, oppure aggiungi un altro tocco centrale.');
        clearTimeout(replay.tapTimer);
        replay.tapTimer = setTimeout(() => previewCard({ suit: replay.selectedSuit, number: replay.taps }), TAP_IDLE_MS);
    }

    function commitCard(card) {
        clearTimeout(replay.tapTimer);
        const target = replay.target;
        replay.pendingCard = null;
        resetCardEntry();

        if (target === 'initial') {
            if (replay.initialCards.some((known) => sameCard(known, card))) return reportError('Questa carta è già nella mano iniziale.');
            replay.initialCards.push(card);
            replay.hand.push(card);
            cue('ok');
            if (replay.initialCards.length < 3) {
                setMessage(`Registrata: ${cardLabel(card)}.`, 'Inserisci la prossima carta iniziale.', cardLabel(card));
                return;
            }
            replay.target = 'trump';
            setMessage('Mano iniziale completa.', 'Inserisci ora la briscola scoperta con lo stesso gesto.', cardLabel(card));
            return;
        }

        if (target === 'trump') {
            if (replay.hand.some((known) => sameCard(known, card))) return reportError('La briscola non può coincidere con una carta della tua mano iniziale.');
            replay.trump = card;
            cue('ok');
            replay.phase = 'opening';
            setMessage('Briscola registrata.', 'Ora: un tocco = inizia l’avversario; tre tocchi = inizi tu.', cardLabel(card));
            return;
        }

        if (target === 'opponent') {
            if (knownPhysicalCards().some((known) => sameCard(known, card))) return reportError('Questa carta è già registrata nel replay.');
            replay.table.push({ card, player_index: 1 });
            replay.opponentHandSize -= 1;
            cue('ok');
            if (replay.table.length === 2) resolveTrick();
            else requestAdvice();
            return;
        }

        if (target === 'player') {
            const index = replay.hand.findIndex((known) => sameCard(known, card));
            if (index === -1) return reportError('Per il replay devi registrare una carta presente nella tua mano.');
            replay.hand.splice(index, 1);
            replay.table.push({ card, player_index: 0 });
            cue('ok');
            if (replay.table.length === 2) resolveTrick();
            else awaitOpponentCard();
            return;
        }

        if (target === 'draw') {
            if (knownPhysicalCards().some((known) => sameCard(known, card))) return reportError('La carta pescata è già comparsa nel replay.');
            replay.hand.push(card);
            cue('ok');
            continueAfterDraw();
        }
    }

    function handleOpeningTap() {
        openingTaps += 1;
        clearTimeout(openingTimer);
        openingTimer = setTimeout(() => {
            const count = openingTaps;
            openingTaps = 0;
            if (count === 1) awaitOpponentCard();
            else if (count === 3) requestAdvice();
            else setMessage('Comando iniziale non riconosciuto.', 'Un tocco se inizia l’avversario; tre tocchi se inizi tu.', '', 'error');
        }, OPENING_IDLE_MS);
        setMessage(`Comando iniziale: ${openingTaps} tocco/i.`, 'Attendi un istante per confermare.');
    }

    function awaitOpponentCard() {
        replay.phase = 'card';
        replay.target = 'opponent';
        resetCardEntry();
        setMessage('Registra la carta giocata dall’avversario.', 'Swipe per il seme, poi valore al centro o in un angolo.');
    }

    function requestAdvice() {
        if (!replay.trump || !replay.hand.length) return finishReplay();
        replay.phase = 'thinking';
        setMessage('Analisi della posizione registrata…', 'PIMC belief 64×10 sta valutando solo le informazioni pubbliche.', '', 'thinking');
        const payload = {
            hand: replay.hand,
            trump_card: replay.trump,
            completed_cards: replay.completed,
            table_cards: replay.table,
            deck_size: replay.deckSize,
            opponent_hand_size: replay.opponentHandSize,
            my_points: replay.points[0],
            opponent_points: replay.points[1],
            first_player: replay.firstPlayer,
        };
        fetch(REPLAY_ADVICE_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        })
            .then(async (response) => {
                const body = await response.json();
                if (!response.ok) throw new Error(body.detail || 'Analisi non disponibile');
                return body;
            })
            .then((body) => {
                replay.recommendation = { suit: body.card.suit, number: body.card.number };
                replay.phase = 'advice';
                cue('ok');
                setAdviceMessage(
                    'La scelta dell’IA per questo momento registrato:',
                    'Carta da giocare.',
                    cardLabel(replay.recommendation),
                    'Tocca per confermare · Swipe per inserire una carta diversa',
                );
            })
            .catch((error) => {
                replay.phase = 'card';
                replay.target = replay.table.length ? 'player' : 'opponent';
                setMessage('Impossibile analizzare il replay.', error.message, '', 'error');
            });
    }

    function registerRecommendedCard() {
        if (!replay.recommendation) return;
        const card = replay.recommendation;
        replay.recommendation = null;
        replay.phase = 'card';
        replay.target = 'player';
        commitCard(card);
    }

    /** Conferma la carta nell'anteprima senza richiedere di inserirla una seconda volta. */
    function confirmPendingCard() {
        if (!replay.pendingCard) return;
        const card = replay.pendingCard;
        replay.pendingCard = null;
        replay.phase = 'card';
        commitCard(card);
    }

    function winnerOfTrick() {
        const [lead, reply] = replay.table;
        const leadTrump = lead.card.suit === replay.trump.suit;
        const replyTrump = reply.card.suit === replay.trump.suit;
        if (replyTrump && !leadTrump) return reply.player_index;
        if (leadTrump && !replyTrump) return lead.player_index;
        if (lead.card.suit !== reply.card.suit) return lead.player_index;
        return STRENGTH[reply.card.number] > STRENGTH[lead.card.number] ? reply.player_index : lead.player_index;
    }

    function resolveTrick() {
        const winner = winnerOfTrick();
        const points = replay.table.reduce((total, entry) => total + POINTS[entry.card.number], 0);
        replay.points[winner] += points;
        replay.completed.push(...replay.table.map((entry) => entry.card));
        replay.table = [];
        replay.firstPlayer = winner;

        // In Briscola a due, le 34 carte del mazzo vengono pescate sempre a coppie.
        if (replay.deckSize > 0) {
            replay.deckSize -= 2;
            replay.opponentHandSize += 1;
            replay.phase = 'card';
            replay.target = 'draw';
            resetCardEntry();
            setMessage(
                winner === 0 ? `Hai preso ${points} punti.` : `L’avversario ha preso ${points} punti.`,
                'Registra ora la carta che hai pescato.',
            );
            return;
        }
        continueAfterDraw();
    }

    function continueAfterDraw() {
        if (!replay.hand.length) return finishReplay();
        if (replay.firstPlayer === 0) requestAdvice();
        else awaitOpponentCard();
    }

    function finishReplay() {
        replay.phase = 'finished';
        setMessage('Partita completata.', 'Ricarica la schermata per iniziare una nuova partita.');
    }

    surface.addEventListener('pointerdown', (event) => {
        event.preventDefault();
        surface.focus({ preventScroll: true });
        pointerStart = { x: event.clientX, y: event.clientY };
        surface.setPointerCapture?.(event.pointerId);
    });

    surface.addEventListener('pointerup', (event) => {
        event.preventDefault();
        if (!pointerStart || replay.phase === 'thinking' || replay.phase === 'finished') return;
        const dx = event.clientX - pointerStart.x;
        const dy = event.clientY - pointerStart.y;
        const isSwipe = Math.hypot(dx, dy) >= SWIPE_DISTANCE_PX;
        const start = pointerStart;
        pointerStart = null;

        if (replay.phase === 'opening') {
            if (isSwipe) setMessage('Qui servono solo tocchi.', 'Un tocco se inizia l’avversario; tre tocchi se inizi tu.', '', 'error');
            else handleOpeningTap();
            return;
        }

        if (replay.phase === 'advice') {
            if (!isSwipe) registerRecommendedCard();
            else startCardEntry('player', SUIT_BY_DIRECTION[directionFromDelta(dx, dy)]);
            return;
        }

        if (replay.phase === 'confirm-card') {
            if (!isSwipe) confirmPendingCard();
            else {
                replay.pendingCard = null;
                startCardEntry(replay.target, SUIT_BY_DIRECTION[directionFromDelta(dx, dy)]);
            }
            return;
        }

        if (replay.phase !== 'card') return;
        if (isSwipe) {
            startCardEntry(replay.target, SUIT_BY_DIRECTION[directionFromDelta(dx, dy)]);
            return;
        }
        onCardTap(start.x, start.y);
    });

    setMessage('Inserisci la prima delle tue tre carte iniziali.', 'Swipe: ↑ denari · ← coppe · → spade · ↓ bastoni.');
})();
