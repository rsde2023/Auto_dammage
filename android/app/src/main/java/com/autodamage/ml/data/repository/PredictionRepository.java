package com.autodamage.ml.data.repository;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Handler;
import android.os.Looper;

import androidx.lifecycle.LiveData;

import com.autodamage.ml.data.api.ApiClient;
import com.autodamage.ml.data.api.ApiService;
import com.autodamage.ml.data.api.HealthResponse;
import com.autodamage.ml.data.api.PredictionResponse;
import com.autodamage.ml.data.db.AppDatabase;
import com.autodamage.ml.data.db.PredictionDao;
import com.autodamage.ml.data.db.PredictionEntity;
import com.autodamage.ml.data.settings.ServerSettings;
import com.autodamage.ml.util.CropMath;
import com.autodamage.ml.util.ErrorMessages;
import com.autodamage.ml.util.ImageUtils;
import com.autodamage.ml.util.ScoresJson;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import retrofit2.Response;

/**
 * Point d'accès unique aux données : API distante (Retrofit) et historique local (Room).
 * Les ViewModels ne connaissent que cette classe ; les Activities/Fragments ne font aucun appel réseau.
 */
public class PredictionRepository {

    public interface Callback<T> {
        void onSuccess(T value);
        void onError(String message);
    }

    private static volatile PredictionRepository instance;

    private final PredictionDao dao;
    private final ServerSettings settings;
    private final File historyDir;
    private final ExecutorService io = Executors.newSingleThreadExecutor();
    private final Handler main = new Handler(Looper.getMainLooper());

    private PredictionRepository(Context context) {
        Context app = context.getApplicationContext();
        dao = AppDatabase.get(app).predictionDao();
        settings = new ServerSettings(app);
        historyDir = new File(app.getFilesDir(), "history");
    }

    public static PredictionRepository get(Context context) {
        if (instance == null) {
            synchronized (PredictionRepository.class) {
                if (instance == null) instance = new PredictionRepository(context);
            }
        }
        return instance;
    }

    public ServerSettings settings() {
        return settings;
    }

    /**
     * Envoie la photo et le cadre de la zone endommagée à /predict, enregistre le résultat dans l'historique
     * et renvoie l'identifiant de l'enregistrement.
     */
    public void analyze(Bitmap photo, int[] bbox, Callback<Long> callback) {
        final String baseUrl = settings.getBaseUrl();
        io.execute(() -> {
            try {
                ApiService api = ApiClient.get(baseUrl);
                RequestBody imageBody = RequestBody.create(ImageUtils.toJpeg(photo, 92), MediaType.get("image/jpeg"));
                MultipartBody.Part imagePart = MultipartBody.Part.createFormData("image", "photo.jpg", imageBody);
                Response<PredictionResponse> response = api.predict(imagePart,
                        text(CropMath.formatNumber(bbox[0])), text(CropMath.formatNumber(bbox[1])),
                        text(CropMath.formatNumber(bbox[2])), text(CropMath.formatNumber(bbox[3]))).execute();
                if (!response.isSuccessful() || response.body() == null) {
                    String err = readError(response.errorBody());
                    post(() -> callback.onError(ErrorMessages.fromHttp(response.code(), err)));
                    return;
                }
                long id = save(response.body(), ImageUtils.crop(photo, bbox));
                post(() -> callback.onSuccess(id));
            } catch (Exception e) {
                post(() -> callback.onError(ErrorMessages.fromThrowable(e, baseUrl)));
            }
        });
    }

    /** Vérifie que le serveur répond et que le modèle est chargé. */
    public void checkHealth(String baseUrl, Callback<HealthResponse> callback) {
        io.execute(() -> {
            try {
                Response<HealthResponse> r = ApiClient.get(baseUrl).health().execute();
                if (r.isSuccessful() && r.body() != null) {
                    post(() -> callback.onSuccess(r.body()));
                } else {
                    post(() -> callback.onError(ErrorMessages.fromHttp(r.code(), null)));
                }
            } catch (Exception e) {
                post(() -> callback.onError(ErrorMessages.fromThrowable(e, baseUrl)));
            }
        });
    }

    public LiveData<List<PredictionEntity>> history() {
        return dao.observeAll();
    }

    public LiveData<PredictionEntity> record(long id) {
        return dao.observe(id);
    }

    public void clearHistory() {
        io.execute(() -> {
            for (String path : dao.thumbnailPaths()) {
                if (path != null) //noinspection ResultOfMethodCallIgnored
                    new File(path).delete();
            }
            dao.deleteAll();
        });
    }

    private long save(PredictionResponse r, Bitmap region) throws IOException {
        long now = System.currentTimeMillis();
        File thumb = new File(historyDir, now + ".jpg");
        ImageUtils.saveJpeg(ImageUtils.scaleDown(region, ImageUtils.THUMBNAIL_SIDE), thumb, 85);
        PredictionEntity e = new PredictionEntity();
        e.createdAt = now;
        e.label = r.label;
        e.labelFr = r.labelFr;
        e.confidence = r.confidence;
        e.uncertain = r.uncertain;
        e.topJson = ScoresJson.toJson(r.top);
        e.model = r.model;
        e.disclaimer = r.disclaimer;
        e.thumbnailPath = thumb.getAbsolutePath();
        return dao.insert(e);
    }

    private static RequestBody text(String value) {
        return RequestBody.create(value, MediaType.get("text/plain"));
    }

    private static String readError(ResponseBody body) {
        if (body == null) return null;
        try {
            return body.string();
        } catch (IOException e) {
            return null;
        }
    }

    private void post(Runnable r) {
        main.post(r);
    }
}
