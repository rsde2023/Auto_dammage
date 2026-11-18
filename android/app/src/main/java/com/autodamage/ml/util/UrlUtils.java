package com.autodamage.ml.util;

import java.util.Locale;

/** Normalisation de l'adresse du serveur saisie par l'utilisateur (Java pur : testable sans téléphone). */
public final class UrlUtils {
    private UrlUtils() { }

    /** "192.168.1.20:8000" -> "http://192.168.1.20:8000/" ; null si vide ou invalide. */
    public static String normalizeBaseUrl(String raw) {
        if (raw == null) return null;
        String url = raw.trim();
        if (url.isEmpty() || url.contains(" ")) return null;
        String lower = url.toLowerCase(Locale.ROOT);
        if (!lower.startsWith("http://") && !lower.startsWith("https://")) {
            url = "http://" + url;
        }
        String host = url.substring(url.indexOf("://") + 3);
        if (host.isEmpty() || host.startsWith("/") || host.startsWith(":")) return null;
        return url.endsWith("/") ? url : url + "/";
    }
}
