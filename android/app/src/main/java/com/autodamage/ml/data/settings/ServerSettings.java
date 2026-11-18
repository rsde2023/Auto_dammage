package com.autodamage.ml.data.settings;

import android.content.Context;
import android.content.SharedPreferences;

import com.autodamage.ml.BuildConfig;
import com.autodamage.ml.util.UrlUtils;

/** Adresse de l'API, modifiable dans l'application (émulateur : http://10.0.2.2:8000/). */
public class ServerSettings {
    private static final String KEY_BASE_URL = "base_url";
    private final SharedPreferences prefs;

    public ServerSettings(Context context) {
        prefs = context.getApplicationContext().getSharedPreferences("settings", Context.MODE_PRIVATE);
    }

    public String getBaseUrl() {
        return prefs.getString(KEY_BASE_URL, BuildConfig.API_BASE_URL);
    }

    /** Enregistre l'adresse normalisée ; retourne null si elle est invalide. */
    public String setBaseUrl(String raw) {
        String url = UrlUtils.normalizeBaseUrl(raw);
        if (url != null) {
            prefs.edit().putString(KEY_BASE_URL, url).apply();
        }
        return url;
    }

    public String getDefaultBaseUrl() {
        return BuildConfig.API_BASE_URL;
    }
}
