package com.autodamage.ml.util;

import java.util.Locale;

/**
 * Conversion du cadre choisi à l'écran en rectangle [x, y, largeur, hauteur] en pixels de la photo envoyée,
 * c'est-à-dire le même format que les bbox COCO utilisées à l'entraînement. Java pur : testable sans téléphone.
 */
public final class CropMath {
    private CropMath() { }

    /**
     * @param left,top,right,bottom cadre normalisé (fractions de la largeur / hauteur, entre 0 et 1)
     * @return {x, y, w, h} en pixels entiers, contenus dans l'image, d'au moins 1 pixel
     */
    public static int[] normalizedToPixels(float left, float top, float right, float bottom,
                                           int imageWidth, int imageHeight) {
        float l = clamp(Math.min(left, right)), r = clamp(Math.max(left, right));
        float t = clamp(Math.min(top, bottom)), b = clamp(Math.max(top, bottom));
        int x0 = Math.round(l * imageWidth), y0 = Math.round(t * imageHeight);
        int x1 = Math.round(r * imageWidth), y1 = Math.round(b * imageHeight);
        x0 = Math.min(x0, imageWidth - 1);
        y0 = Math.min(y0, imageHeight - 1);
        int w = Math.max(1, Math.min(x1, imageWidth) - x0);
        int h = Math.max(1, Math.min(y1, imageHeight) - y0);
        return new int[]{x0, y0, w, h};
    }

    /** Nombre envoyé à l'API : toujours avec un point décimal (en français, String.valueOf donnerait parfois une virgule). */
    public static String formatNumber(double value) {
        return String.format(Locale.US, "%.2f", value);
    }

    private static float clamp(float v) {
        return Math.max(0f, Math.min(1f, v));
    }
}
