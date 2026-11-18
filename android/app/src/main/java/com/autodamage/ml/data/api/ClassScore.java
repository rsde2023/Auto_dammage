package com.autodamage.ml.data.api;

import com.google.gson.annotations.SerializedName;

/** Une classe candidate et sa probabilité estimée. */
public class ClassScore {
    public String label;
    @SerializedName("label_fr") public String labelFr;
    public double probability;

    public ClassScore() { }

    public ClassScore(String label, String labelFr, double probability) {
        this.label = label;
        this.labelFr = labelFr;
        this.probability = probability;
    }
}
