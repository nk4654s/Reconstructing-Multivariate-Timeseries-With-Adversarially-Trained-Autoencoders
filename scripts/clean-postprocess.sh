#!/bin/sh
DIRECTORY_RAW="dataset-2021-6m-raw"
DIRECTORY="dataset-2021-6m"

pip install -e .

if [ ! -d "$DIRECTORY_RAW" ]; then
    echo "$DIRECTORY_RAW does not exist. Downloading dataset..."
    flux-pre --folder "$DIRECTORY" --machine-keep-state on --interval 2021-09-01 2022-03-31 --overwrite
    cp -r "$DIRECTORY" "$DIRECTORY_RAW"
else
    echo "$DIRECTORY_RAW does exist. Restoring backup..."
    rm -rf "$DIRECTORY"
    cp -r "$DIRECTORY_RAW" "$DIRECTORY"
fi

REMOVE_COLUMNS="\"ts,Automatenfall/SpannungGestört,Laeuft,Nachlauf,NiveauschalterTriebwerksölLS74504,NothaltVD2Betätigt,PositionAustrittseitigeSicherheitsAbsperrarmaturSSV74369Auf,PositionAustrittseitigeSicherheitsAbsperrarmaturSSV74369Zu,PositionAustrittsseitigesSicherheitsentlüftungsventilSE74368Auf,PositionAustrittsseitigesSicherheitsentlüftungsventilSE74368Zu,PositionBypassregelventilPCV74381Auf,PositionBypassregelventilPCV74381Zu,PositionEintrittseitigeSicherheitsAbsperrarmaturSSV74306Auf,PositionEintrittseitigeSicherheitsAbsperrarmaturSSV74306Zu,PositionEintrittsseitigesSicherheitsentlüftungsventilSE74307Auf,PositionEintrittsseitigesSicherheitsentlüftungsventilSE74307Zu,Sammelwarnung,Startbereit,VD2StörungGekommen,VD2StörungGekommenQuittiert,VerdichterAbschaltungMitEntspannung,VerdichterAbschaltungOhneEntspannung,VerdichterFern,VerdichterOrt,Vorlauf\""
OUTLIER_IGNORE="\"Aussentemperatur (°C),Betriebsstunden (h),KühlwassertemperaturnachKühlerTETIT74422 (°C),PackungsleckageFIT74389 (Nm3h),KühlwasserdrucknachPumpePIT74425 (bar),PositionBypassregelventilPCV74381 (%),SaugtemperaturErsteStufeTETIT74328 (°C),SaugdruckErsteStufePIT74319 (bar),KompressordrehzahlSE74201 (rpm),ÖltemperaturVorlaufTriebwerkTETIT74538 (°C),Motorlager (°C),KühlwasserdurchflussmengeFIT74423 (m^3/min)\""
flux-post --folder "$DIRECTORY" --num-workers 12 --weather-merge outer --weather-interpolate True  --normalize-method minmax --remove-columns "$REMOVE_COLUMNS" --mono-enable False --outlier-sigma 6 --outlier-ignore "$OUTLIER_IGNORE"
