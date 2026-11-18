package com.autodamage.ml.data.api;

import java.util.concurrent.TimeUnit;

import okhttp3.OkHttpClient;
import okhttp3.logging.HttpLoggingInterceptor;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

/** Construit le client Retrofit ; reconstruit seulement si l'adresse du serveur change. */
public final class ApiClient {
    private static ApiService service;
    private static String currentBaseUrl;

    private ApiClient() { }

    public static synchronized ApiService get(String baseUrl) {
        if (service == null || !baseUrl.equals(currentBaseUrl)) {
            HttpLoggingInterceptor logging = new HttpLoggingInterceptor();
            logging.setLevel(HttpLoggingInterceptor.Level.BASIC);
            OkHttpClient http = new OkHttpClient.Builder()
                    .connectTimeout(10, TimeUnit.SECONDS)
                    .readTimeout(30, TimeUnit.SECONDS)
                    .writeTimeout(30, TimeUnit.SECONDS)
                    .addInterceptor(logging)
                    .build();
            service = new Retrofit.Builder()
                    .baseUrl(baseUrl)
                    .client(http)
                    .addConverterFactory(GsonConverterFactory.create())
                    .build()
                    .create(ApiService.class);
            currentBaseUrl = baseUrl;
        }
        return service;
    }
}
