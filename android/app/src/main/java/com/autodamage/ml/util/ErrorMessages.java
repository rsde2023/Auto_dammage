package com.autodamage.ml.util;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.net.ConnectException;
import java.net.SocketTimeoutException;
import java.net.UnknownHostException;

/** Traduit les erreurs réseau / HTTP en messages compréhensibles par l'utilisateur. */
public final class ErrorMessages {
    private ErrorMessages() { }

    public static String fromHttp(int code, String errorBody) {
        String detail = extractDetail(errorBody);
        switch (code) {
            case 503:
                return "Le serveur fonctionne mais le modèle n'est pas chargé. Sur le PC : python -m src.models.train";
            case 413:
                return "Image trop lourde pour le serveur.";
            case 400:
            case 422:
                return detail != null ? detail : "Requête refusée par le serveur (image ou cadre invalide).";
            default:
                return "Erreur du serveur (code " + code + ")" + (detail != null ? " : " + detail : ".");
        }
    }

    public static String fromThrowable(Throwable t, String baseUrl) {
        if (t instanceof ConnectException || t instanceof UnknownHostException) {
            return "Impossible de joindre le serveur " + baseUrl
                    + ". Vérifiez que l'API est lancée (uvicorn) et l'adresse dans le menu ⋮.";
        }
        if (t instanceof SocketTimeoutException) {
            return "Le serveur " + baseUrl + " ne répond pas (délai dépassé).";
        }
        if (t instanceof IOException) {
            return "Erreur réseau : " + t.getMessage();
        }
        return "Erreur inattendue : " + t.getMessage();
    }

    /** FastAPI renvoie {"detail": "message"} (ou une liste pour les erreurs de validation). */
    static String extractDetail(String body) {
        if (body == null || body.isEmpty()) return null;
        try {
            JsonElement root = JsonParser.parseString(body);
            if (!root.isJsonObject()) return null;
            JsonObject obj = root.getAsJsonObject();
            JsonElement detail = obj.get("detail");
            if (detail != null && detail.isJsonPrimitive()) return detail.getAsString();
        } catch (RuntimeException ignored) {
            // corps non JSON : pas de détail
        }
        return null;
    }
}
