# Metódy hodnotenia kvality vysvetliteľných modelov umelej inteligencie

Tento repozitár obsahuje zdrojové kódy k diplomovej práci zameranej na hodnotenie kvality vysvetlení XAI, vrátane objektívnych aj subjektívnych metrík.



## Štruktúra repozitára

Repozitár je rozdelený do nasledovných priečinkov:

- **00_data**: Obsahuje dáta použité v experimentoch: pôvodný dataset, nové rozdelenie dát, vzorka dát (test sample), odpovede respondentov, popisy obrázkov.
- **01_caption_creation**: Proces generovania textových popisov k obrázkom
- **02_bert_model**: Implementácia textového modelu BERT
- **03_hateclipper_model**: Multimodálny model pracujúci s textom a obrázkami: kombinácia vizuálnych a textových vstupov, klasifikácia na základe oboch modalít.
- **04_bert_xai**: Výpočet objektívnych metrík pre metódy vysvetliteľnosti aplikované na textový model
- **05_hateclipper_xai**: Výpočet objektívnych metrík pre metódy vysvetliteľnosti aplikované na multimodálny model
- **06_web_subjective_metrics**: Webová aplikácia pre zber používateľských hodnotení vysvetlení.
- **07_MCDM**: Implementácia upravenej verzie tretej stránky aplikácie pre multikriteriálne rozhodovanie
