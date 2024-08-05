#!/bin/sh
DIRECTORY_RAW="dataset-2021-23-raw"
DIRECTORY="dataset-2021-23"

pip install -e .

if [ ! -d "$DIRECTORY_RAW" ]; then
    echo "$DIRECTORY_RAW does not exist. Downloading dataset..."
    flux-pre --folder "$DIRECTORY" --machine-keep-state on --interval 2021-09-01 2022-08-31 --overwrite
    cp -r "$DIRECTORY" "$DIRECTORY_RAW"
else
    echo "$DIRECTORY_RAW does exist. Restoring backup..."
    rm -rf "$DIRECTORY"
    cp -r "$DIRECTORY_RAW" "$DIRECTORY"
fi

REMOVE_COLUMNS="\"ts Automatenfall/SpannungGestört Laeuft Nachlauf NiveauschalterTriebwerksölLS74504 NothaltVD2Betätigt PositionAustrittseitigeSicherheitsAbsperrarmaturSSV74369Auf PositionAustrittseitigeSicherheitsAbsperrarmaturSSV74369Zu PositionAustrittsseitigesSicherheitsentlüftungsventilSE74368Auf PositionAustrittsseitigesSicherheitsentlüftungsventilSE74368Zu PositionBypassregelventilPCV74381Auf PositionBypassregelventilPCV74381Zu PositionEintrittseitigeSicherheitsAbsperrarmaturSSV74306Auf PositionEintrittseitigeSicherheitsAbsperrarmaturSSV74306Zu PositionEintrittsseitigesSicherheitsentlüftungsventilSE74307Auf PositionEintrittsseitigesSicherheitsentlüftungsventilSE74307Zu Sammelwarnung Startbereit VD2StörungGekommen VD2StörungGekommenQuittiert VerdichterAbschaltungMitEntspannung VerdichterAbschaltungOhneEntspannung VerdichterFern VerdichterOrt Vorlauf\""
flux-post --folder "$DIRECTORY" --num-workers 12 --weather-merge outer --weather-interpolate True  --normalize-method minmax --remove-columns "$REMOVE_COLUMNS" --mono-enable False
