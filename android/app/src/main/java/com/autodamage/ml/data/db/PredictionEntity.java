package com.autodamage.ml.data.db;

import androidx.annotation.NonNull;
import androidx.room.Entity;
import androidx.room.PrimaryKey;

/** Une analyse enregistrée dans l'historique local (table Room « predictions »). */
@Entity(tableName = "predictions")
public class PredictionEntity {
    @PrimaryKey(autoGenerate = true)
    public long id;
    public long createdAt;          // millisecondes depuis 1970
    @NonNull public String label = "";
    @NonNull public String labelFr = "";
    public double confidence;
    public boolean uncertain;
    @NonNull public String topJson = "[]";   // liste des 3 classes les plus probables, en JSON
    public String model;
    public String disclaimer;
    public String thumbnailPath;    // vignette de la zone analysée (stockage interne de l'application)
}
