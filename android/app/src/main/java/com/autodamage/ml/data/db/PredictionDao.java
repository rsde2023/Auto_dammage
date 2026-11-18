package com.autodamage.ml.data.db;

import androidx.lifecycle.LiveData;
import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.Query;

import java.util.List;

@Dao
public interface PredictionDao {
    @Insert
    long insert(PredictionEntity entity);

    @Query("SELECT * FROM predictions ORDER BY createdAt DESC")
    LiveData<List<PredictionEntity>> observeAll();

    @Query("SELECT * FROM predictions WHERE id = :id")
    LiveData<PredictionEntity> observe(long id);

    @Query("SELECT thumbnailPath FROM predictions")
    List<String> thumbnailPaths();

    @Query("DELETE FROM predictions")
    void deleteAll();
}
