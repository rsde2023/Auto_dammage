package com.autodamage.ml.ui;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.navigation.NavController;
import androidx.navigation.fragment.NavHostFragment;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;

import com.autodamage.ml.R;
import com.autodamage.ml.data.api.HealthResponse;
import com.autodamage.ml.data.repository.PredictionRepository;
import com.autodamage.ml.data.settings.ServerSettings;
import com.autodamage.ml.databinding.ActivityMainBinding;
import com.autodamage.ml.databinding.DialogServerBinding;
import com.autodamage.ml.util.UrlUtils;
import com.google.android.material.dialog.MaterialAlertDialogBuilder;

/** Activité unique : barre d'outils, navigation du bas et réglage de l'adresse du serveur. */
public class MainActivity extends AppCompatActivity {

    private NavController navController;
    private AppBarConfiguration appBarConfiguration;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        ActivityMainBinding binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        setSupportActionBar(binding.toolbar);

        NavHostFragment host = (NavHostFragment) getSupportFragmentManager().findFragmentById(R.id.nav_host);
        navController = host.getNavController();
        appBarConfiguration = new AppBarConfiguration.Builder(R.id.analyzeFragment, R.id.historyFragment).build();
        NavigationUI.setupActionBarWithNavController(this, navController, appBarConfiguration);
        NavigationUI.setupWithNavController(binding.bottomNav, navController);
        navController.addOnDestinationChangedListener((c, dest, args) ->
                binding.bottomNav.setVisibility(dest.getId() == R.id.resultFragment ? View.GONE : View.VISIBLE));
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        if (item.getItemId() == R.id.action_server) {
            showServerDialog();
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    @Override
    public boolean onSupportNavigateUp() {
        return NavigationUI.navigateUp(navController, appBarConfiguration) || super.onSupportNavigateUp();
    }

    /** Adresse de l'API : 10.0.2.2:8000 sur l'émulateur, IP du PC sur le Wi-Fi pour un vrai téléphone. */
    private void showServerDialog() {
        PredictionRepository repo = PredictionRepository.get(this);
        ServerSettings settings = repo.settings();
        DialogServerBinding d = DialogServerBinding.inflate(LayoutInflater.from(this));
        d.editUrl.setText(settings.getBaseUrl());
        d.textHint.setText(getString(R.string.server_hint, settings.getDefaultBaseUrl()));

        AlertDialog dialog = new MaterialAlertDialogBuilder(this)
                .setTitle(R.string.server_title)
                .setView(d.getRoot())
                .setNegativeButton(R.string.cancel, null)
                .setNeutralButton(R.string.server_test, null)
                .setPositiveButton(R.string.save, null)
                .create();
        dialog.setOnShowListener(x -> {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener(v -> {
                if (settings.setBaseUrl(String.valueOf(d.editUrl.getText())) == null) {
                    d.inputUrl.setError(getString(R.string.server_invalid));
                } else {
                    dialog.dismiss();
                }
            });
            dialog.getButton(AlertDialog.BUTTON_NEUTRAL).setOnClickListener(v -> {
                String url = UrlUtils.normalizeBaseUrl(String.valueOf(d.editUrl.getText()));
                if (url == null) {
                    d.inputUrl.setError(getString(R.string.server_invalid));
                    return;
                }
                d.inputUrl.setError(null);
                d.textStatus.setText(R.string.server_testing);
                repo.checkHealth(url, new PredictionRepository.Callback<HealthResponse>() {
                    @Override
                    public void onSuccess(HealthResponse h) {
                        d.textStatus.setText(h.modelLoaded
                                ? getString(R.string.server_ok, h.model)
                                : getString(R.string.server_no_model));
                    }

                    @Override
                    public void onError(String message) {
                        d.textStatus.setText(message);
                    }
                });
            });
        });
        dialog.show();
    }
}
