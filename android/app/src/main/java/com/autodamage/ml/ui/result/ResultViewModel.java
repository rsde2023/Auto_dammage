package com.autodamage.ml.ui.result;

import android.app.Application;

import androidx.annotation.NonNull;
import androidx.lifecycle.AndroidViewModel;
import androidx.lifecycle.LiveData;

import com.autodamage.ml.data.db.PredictionEntity;
import com.autodamage.ml.data.repository.PredictionRepository;

/** Fournit l'analyse à afficher (lue dans l'historique Room). */
public class ResultViewModel extends AndroidViewModel {
    private final PredictionRepository repository;
    private LiveData<PredictionEntity> record;
    private long recordId = -1;

    public ResultViewModel(@NonNull Application app) {
        super(app);
        repository = PredictionRepository.get(app);
    }

    public LiveData<PredictionEntity> record(long id) {
        if (record == null || id != recordId) {
            recordId = id;
            record = repository.record(id);
        }
        return record;
    }
}
