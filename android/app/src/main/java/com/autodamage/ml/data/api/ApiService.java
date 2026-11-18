package com.autodamage.ml.data.api;

import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.http.GET;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;

/** Points d'accès de l'API FastAPI (src/api/main.py). */
public interface ApiService {

    @GET("health")
    Call<HealthResponse> health();

    /** Photo + rectangle de la zone endommagée, en pixels de la photo envoyée. */
    @Multipart
    @POST("predict")
    Call<PredictionResponse> predict(@Part MultipartBody.Part image,
                                     @Part("bbox_x") RequestBody bboxX,
                                     @Part("bbox_y") RequestBody bboxY,
                                     @Part("bbox_w") RequestBody bboxW,
                                     @Part("bbox_h") RequestBody bboxH);
}
