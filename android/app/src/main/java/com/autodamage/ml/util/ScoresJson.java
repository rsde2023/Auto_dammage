package com.autodamage.ml.util;

import com.autodamage.ml.data.api.ClassScore;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;

/** Stockage de la liste des classes candidates dans une colonne texte de Room. */
public final class ScoresJson {
    private static final Gson GSON = new Gson();
    private static final Type TYPE = new TypeToken<List<ClassScore>>() { }.getType();

    private ScoresJson() { }

    public static String toJson(List<ClassScore> scores) {
        return GSON.toJson(scores == null ? new ArrayList<ClassScore>() : scores, TYPE);
    }

    public static List<ClassScore> fromJson(String json) {
        if (json == null || json.isEmpty()) return new ArrayList<>();
        List<ClassScore> out = GSON.fromJson(json, TYPE);
        return out == null ? new ArrayList<>() : out;
    }
}
