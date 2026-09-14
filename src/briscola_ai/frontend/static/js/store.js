/**
 * Mini store (pub/sub) in stile "framework".
 *
 * Obiettivo:
 * - avere una singola sorgente di verità per lo stato UI
 * - rendere il rendering deterministico: ogni update passa dallo store
 *
 * Nota: non usiamo moduli ES o bundler; esponiamo `Store` come globale.
 */

const Store = (() => {
    /**
     * Crea uno store con stato e subscription.
     * @param {Object} initialState
     */
    const create = (initialState = {}) => {
        let state = { ...initialState };
        const listeners = new Set();

        const getState = () => state;

        /**
         * Applica una patch shallow e notifica i listener.
         * @param {Object} patch
         */
        const setState = (patch) => {
            state = { ...state, ...patch };
            listeners.forEach((listener) => listener(state));
        };

        /**
         * Subscribe (immediato) e ritorna un unsubscribe.
         * @param {(state: Object) => void} listener
         */
        const subscribe = (listener) => {
            listeners.add(listener);
            listener(state);
            return () => listeners.delete(listener);
        };

        return { getState, setState, subscribe };
    };

    return { create };
})();

