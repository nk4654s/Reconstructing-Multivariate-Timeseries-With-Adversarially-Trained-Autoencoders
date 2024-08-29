#!/bin/sh
if [ -z "$1" ]; then
    echo "Please specify a dataset name."
    exit 1
fi

DIRECTORY_RAW="$1-raw"
DIRECTORY="$1"

pip install -e .

if [ ! -d "$DIRECTORY_RAW" ]; then
    if [ -z "$2" ]; then
        echo "Please specify a start date (YYYY-MM-DD)."
        exit 1
    fi

    if [ -z "$3" ]; then
        echo "Please specify an end date (YYYY-MM-DD)."
        exit 1
    fi
    echo "$DIRECTORY_RAW does not exist. Downloading dataset..."
    flux-pre --folder "$DIRECTORY" --machine-keep-state on --interval "$2" "$3" --position-remove-timespan 1800 --overwrite
    cp -r "$DIRECTORY" "$DIRECTORY_RAW"
else
    echo "$DIRECTORY_RAW does exist. Restoring backup..."
    rm -rf "$DIRECTORY"
    cp -r "$DIRECTORY_RAW" "$DIRECTORY"
fi

REMOVE_COLUMNS="\"ts,Betriebsstunden (h),Automatenfall/SpannungGestört,Laeuft,Nachlauf,NiveauschalterTriebwerksölLS74504,NothaltVD2Betätigt,PositionAustrittseitigeSicherheitsAbsperrarmaturSSV74369Auf,PositionAustrittseitigeSicherheitsAbsperrarmaturSSV74369Zu,PositionAustrittsseitigesSicherheitsentlüftungsventilSE74368Auf,PositionAustrittsseitigesSicherheitsentlüftungsventilSE74368Zu,PositionBypassregelventilPCV74381Auf,PositionBypassregelventilPCV74381Zu,PositionEintrittseitigeSicherheitsAbsperrarmaturSSV74306Auf,PositionEintrittseitigeSicherheitsAbsperrarmaturSSV74306Zu,PositionEintrittsseitigesSicherheitsentlüftungsventilSE74307Auf,PositionEintrittsseitigesSicherheitsentlüftungsventilSE74307Zu,Sammelwarnung,Startbereit,VD2StörungGekommen,VD2StörungGekommenQuittiert,VerdichterAbschaltungMitEntspannung,VerdichterAbschaltungOhneEntspannung,VerdichterFern,VerdichterOrt,Vorlauf,Biomethandurchfluss (m^3/min),Motor (A),Motor (kW),PositionBypassregelventilPCV74381 (%)\""
OUTLIER_IGNORE="\"Aussentemperatur (°C),KühlwassertemperaturnachKühlerTETIT74422 (°C),PackungsleckageFIT74389 (Nm3h),KühlwasserdrucknachPumpePIT74425 (bar),SaugtemperaturErsteStufeTETIT74328 (°C),SaugdruckErsteStufePIT74319 (bar),KompressordrehzahlSE74201 (rpm),ÖltemperaturVorlaufTriebwerkTETIT74538 (°C),Motorlager (°C),KühlwasserdurchflussmengeFIT74423 (m^3/min)\""
flux-post --folder "$DIRECTORY" --weather-enable False --num-workers 12 --remove-columns "$REMOVE_COLUMNS" --mono-enable False --outlier-enable False --outlier-sigma 6 --outlier-ignore "$OUTLIER_IGNORE" --normalize-method minmax --normalize-from-other dataset-cat-green
# --weather-merge outer --weather-interpolate True