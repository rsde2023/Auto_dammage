package com.autodamage.ml.data.api;

import com.google.gson.annotations.SerializedName;

import java.util.List;

/** Réponse JSON de GET /health. */
public class HealthResponse {
    public String status;
    @SerializedName("model_loaded") public boolean modelLoaded;
    public String model;
    public List<String> classes;
}
