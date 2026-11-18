package com.autodamage.ml.ui.analyze;

import android.app.Application;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.net.Uri;

import androidx.annotation.NonNull;
import androidx.lifecycle.AndroidViewModel;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.autodamage.ml.data.repository.PredictionRepository;
import com.autodamage.ml.util.CropMath;
import com.autodamage.ml.util.ImageUtils;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Garde la photo choisie et le cadre pendant les rotations d'écran, et lance l'analyse via le Repository.
 */
public class AnalyzeViewModel extends AndroidViewModel {
    private static final RectF DEFAULT_CROP = new RectF(0.2f, 0.2f, 0.8f, 0.8f);

    private final PredictionRepository repository;
    private final ExecutorService imageExecutor = Executors.newSingleThreadExecutor();
    private final MutableLiveData<Bitmap> photo = new MutableLiveData<>();
    private final MutableLiveData<AnalyzeState> state = new MutableLiveData<>(AnalyzeState.idle());
    private RectF crop = new RectF(DEFAULT_CROP);

    public AnalyzeViewModel(@NonNull Application app) {
        super(app);
        repository = PredictionRepository.get(app);
    }

    public LiveData<Bitmap> photo() { return photo; }
    public LiveData<AnalyzeState> state() { return state; }
    public RectF crop() { return new RectF(crop); }
    public void setCrop(RectF normalized) { crop = new RectF(normalized); }

    public void loadPhoto(Uri uri) {
        state.setValue(AnalyzeState.loadingPhoto());
        imageExecutor.execute(() -> {
            try {
                Bitmap bmp = ImageUtils.loadUpright(getApplication().getContentResolver(), uri, ImageUtils.MAX_UPLOAD_SIDE);
                crop = new RectF(DEFAULT_CROP);
                photo.postValue(bmp);
                state.postValue(AnalyzeState.idle());
            } catch (Exception e) {
                state.postValue(AnalyzeState.error("Impossible de lire cette image : " + e.getMessage()));
            }
        });
    }

    public void analyze() {
        Bitmap bmp = photo.getValue();
        if (bmp == null || state.getValue() != null && state.getValue().isBusy()) return;
        int[] bbox = CropMath.normalizedToPixels(crop.left, crop.top, crop.right, crop.bottom, bmp.getWidth(), bmp.getHeight());
        state.setValue(AnalyzeState.analyzing());
        repository.analyze(bmp, bbox, new PredictionRepository.Callback<Long>() {
            @Override public void onSuccess(Long id) { state.setValue(AnalyzeState.success(id)); }
            @Override public void onError(String message) { state.setValue(AnalyzeState.error(message)); }
        });
    }

    /** À appeler une fois la navigation ou le message affiché, pour ne pas les répéter après une rotation. */
    public void consumeEvent() {
        state.setValue(AnalyzeState.idle());
    }

    public void clearPhoto() {
        photo.setValue(null);
        crop = new RectF(DEFAULT_CROP);
        state.setValue(AnalyzeState.idle());
    }

    @Override
    protected void onCleared() {
        imageExecutor.shutdown();
    }
}
