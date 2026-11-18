package com.autodamage.ml.data.api;

import com.google.gson.annotations.SerializedName;

import java.util.List;

/** Réponse JSON de POST /predict (voir src/api/main.py, classe PredictionResponse). */
public class PredictionResponse {
    public String label;
    @SerializedName("label_fr") public String labelFr;
    public double confidence;
    public boolean uncertain;
    public List<ClassScore> top;
    public String model;
    @SerializedName("image_width") public int imageWidth;
    @SerializedName("image_height") public int imageHeight;
    @SerializedName("processing_ms") public int processingMs;
    public String disclaimer;
}
