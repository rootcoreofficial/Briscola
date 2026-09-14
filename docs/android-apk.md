# APK dell'assistente Briscola

La app Android usa [Capacitor](https://capacitorjs.com/docs): il contenitore nativo include nel suo WebView la
schermata touch pensata per una partita fisica. L'analisi della carta resta sul backend pubblico, quindi l'APK
necessita di una connessione Internet per ottenere il consiglio; non incorpora ne' espone il modello IA sul telefono.

Nella build Android le richieste al backend passano attraverso il client HTTPS nativo di Capacitor. Questo evita che
la WebView sia bloccata da CORS, mantenendo invece il comportamento HTTP standard quando la stessa pagina e' aperta
nel browser.

## Prima build

Su un computer con Android Studio, Android SDK e una JDK supportata dalla versione di Android Gradle Plugin
installata da Capacitor, dalla radice del repository eseguire:

```powershell
npm install
npm run android:add
npm run android:sync
New-Item -ItemType Directory -Force logs | Out-Null
npm run android:debug *>&1 | Tee-Object -FilePath logs\android-debug.log
```

L'APK di debug sara' in `android/app/build/outputs/apk/debug/app-debug.apk`. Per installarlo su un telefono
collegato via USB e con Debug USB abilitato:

```powershell
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
```

Nel log cercare `BUILD SUCCESSFUL`; in caso contrario, le ultime righe contengono il requisito Android SDK o JDK
mancante. La prima build puo' scaricare Gradle e dipendenze Android e richiedere alcuni minuti.

## Aggiornamenti

La UI e' inclusa nell'APK. Dopo una modifica alla pagina `/replay` occorre eseguire `npm run android:sync` e
ricreare l'APK; il server viene contattato solo per il suggerimento IA. Ricreare l'APK serve anche per cambiare
icona, nome, permessi oppure per pubblicarlo su uno store.

Non distribuire l'APK di debug: prima della pubblicazione va creato un keystore di rilascio e prodotta una
build firmata/AAB dall'Android Studio project generato.
