package com.autodamage.ml.ui.history;

import android.app.Application;

import androidx.annotation.NonNull;
import androidx.lifecycle.AndroidViewModel;
import androidx.lifecycle.LiveData;

import com.autodamage.ml.data.db.PredictionEntity;
import com.autodamage.ml.data.repository.PredictionRepository;

import java.util.List;

public class HistoryViewModel extends AndroidViewModel {
    private final PredictionRepository repository;
    private final LiveData<List<PredictionEntity>> history;

    public HistoryViewModel(@NonNull Application app) {
        super(app);
        repository = PredictionRepository.get(app);
        history = repository.history();
    }

    public LiveData<List<PredictionEntity>> history() { return history; }

    public void clear() { repository.clearHistory(); }
}
