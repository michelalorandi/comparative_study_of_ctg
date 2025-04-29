#!/bin/bash

echo "Start download models"
cd  ./scripts/src/models/gedi
mkdir ./pretrained_models
cd ./pretrained_models

echo "Start dowloading Topic model"
wget https://storage.googleapis.com/sfr-gedi-data/gedi_topic.zip
unzip gedi_topic.zip
rm gedi_topic.zip
echo "End downloading Topic model"

echo "Start dowloading Sentiment model"
wget https://storage.googleapis.com/sfr-gedi-data/gedi_sentiment.zip
unzip gedi_sentiment.zip
rm gedi_sentiment.zip
echo "End download Sentiment model"

echo "End Dowload models"


