package com.autodamage.ml.util;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.autodamage.ml.data.api.ClassScore;

import org.junit.Test;

import java.net.ConnectException;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/** Tests unitaires (JVM, sans téléphone) : ./gradlew testDebugUnitTest */
public class UtilsTest {

    @Test
    public void baseUrlIsNormalized() {
        assertEquals("http://10.0.2.2:8000/", UrlUtils.normalizeBaseUrl("http://10.0.2.2:8000"));
        assertEquals("http://192.168.1.20:8000/", UrlUtils.normalizeBaseUrl("  192.168.1.20:8000 "));
        assertEquals("https://api.example.org/", UrlUtils.normalizeBaseUrl("https://api.example.org/"));
        assertNull(UrlUtils.normalizeBaseUrl(""));
        assertNull(UrlUtils.normalizeBaseUrl("http://"));
        assertNull(UrlUtils.normalizeBaseUrl("http://mon serveur"));
    }

    @Test
    public void cropIsConvertedToCocoPixels() {
        // cadre de 25 % à 75 % d'une photo 1000 x 800 -> x=250, y=200, w=500, h=400
        assertArrayEquals(new int[]{250, 200, 500, 400},
                CropMath.normalizedToPixels(0.25f, 0.25f, 0.75f, 0.75f, 1000, 800));
    }

    @Test
    public void cropIsClampedAndNeverEmpty() {
        assertArrayEquals(new int[]{0, 0, 100, 50}, CropMath.normalizedToPixels(-0.2f, -1f, 1.5f, 2f, 100, 50));
        int[] tiny = CropMath.normalizedToPixels(1f, 1f, 1f, 1f, 100, 50);
        assertTrue(tiny[2] >= 1 && tiny[3] >= 1 && tiny[0] + tiny[2] <= 100 && tiny[1] + tiny[3] <= 50);
        // coins inversés : même résultat
        assertArrayEquals(CropMath.normalizedToPixels(0.1f, 0.2f, 0.6f, 0.9f, 640, 480),
                CropMath.normalizedToPixels(0.6f, 0.9f, 0.1f, 0.2f, 640, 480));
    }

    @Test
    public void numbersSentToApiAlwaysUseADot() {
        Locale previous = Locale.getDefault();
        try {
            Locale.setDefault(Locale.FRANCE);
            assertEquals("12.50", CropMath.formatNumber(12.5));
        } finally {
            Locale.setDefault(previous);
        }
    }

    @Test
    public void fastApiErrorDetailIsShown() {
        assertEquals("bbox incomplète", ErrorMessages.fromHttp(422, "{\"detail\":\"bbox incomplète\"}"));
        assertTrue(ErrorMessages.fromHttp(422, "{\"detail\":[{\"msg\":\"x\"}]}").contains("Requête refusée"));
        assertTrue(ErrorMessages.fromHttp(503, null).contains("src.models.train"));
        assertTrue(ErrorMessages.fromHttp(500, "pas du json").contains("500"));
        assertTrue(ErrorMessages.fromThrowable(new ConnectException("refused"), "http://10.0.2.2:8000/")
                .contains("10.0.2.2"));
    }

    @Test
    public void scoresSurviveJsonRoundTrip() {
        List<ClassScore> scores = Arrays.asList(new ClassScore("dent", "Bosse", 0.7), new ClassScore("scratch", "Rayure", 0.2));
        List<ClassScore> back = ScoresJson.fromJson(ScoresJson.toJson(scores));
        assertEquals(2, back.size());
        assertEquals("Bosse", back.get(0).labelFr);
        assertEquals(0.2, back.get(1).probability, 1e-9);
        assertTrue(ScoresJson.fromJson(null).isEmpty());
    }
}
