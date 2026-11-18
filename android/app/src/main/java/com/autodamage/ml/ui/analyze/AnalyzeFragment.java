package com.autodamage.ml.ui.analyze;

import android.content.ActivityNotFoundException;
import android.net.Uri;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.content.FileProvider;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.navigation.fragment.NavHostFragment;

import com.autodamage.ml.R;
import com.autodamage.ml.databinding.FragmentAnalyzeBinding;
import com.autodamage.ml.ui.result.ResultFragment;
import com.google.android.material.snackbar.Snackbar;

import java.io.File;

/** Écran 1 : choisir une photo, encadrer la zone endommagée, lancer l'analyse. Aucun calcul ML ici. */
public class AnalyzeFragment extends Fragment {

    private FragmentAnalyzeBinding binding;
    private AnalyzeViewModel viewModel;
    private Uri pendingCameraUri;

    private final ActivityResultLauncher<String> pickImage =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) viewModel.loadPhoto(uri);
            });

    private final ActivityResultLauncher<Uri> takePicture =
            registerForActivityResult(new ActivityResultContracts.TakePicture(), saved -> {
                if (Boolean.TRUE.equals(saved) && pendingCameraUri != null) viewModel.loadPhoto(pendingCameraUri);
            });

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle state) {
        binding = FragmentAnalyzeBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        viewModel = new ViewModelProvider(this).get(AnalyzeViewModel.class);
        if (savedInstanceState != null) {
            String uri = savedInstanceState.getString("camera_uri");
            if (uri != null) pendingCameraUri = Uri.parse(uri);
        }

        binding.buttonGallery.setOnClickListener(v -> pickImage.launch("image/*"));
        binding.buttonCamera.setOnClickListener(v -> openCamera());
        binding.buttonChangePhoto.setOnClickListener(v -> viewModel.clearPhoto());
        binding.buttonAnalyze.setOnClickListener(v -> viewModel.analyze());
        binding.cropView.setOnCropChangeListener(viewModel::setCrop);

        viewModel.photo().observe(getViewLifecycleOwner(), bmp -> {
            boolean hasPhoto = bmp != null;
            binding.groupEmpty.setVisibility(hasPhoto ? View.GONE : View.VISIBLE);
            binding.groupCrop.setVisibility(hasPhoto ? View.VISIBLE : View.GONE);
            binding.cropView.setImageBitmap(bmp);
            if (hasPhoto) binding.cropView.setCropNormalized(viewModel.crop());
        });

        viewModel.state().observe(getViewLifecycleOwner(), s -> {
            binding.progress.setVisibility(s.isBusy() ? View.VISIBLE : View.GONE);
            binding.buttonAnalyze.setEnabled(!s.isBusy());
            binding.buttonAnalyze.setText(s.status == AnalyzeState.Status.ANALYZING
                    ? R.string.analyzing : R.string.analyze);
            if (s.status == AnalyzeState.Status.SUCCESS) {
                viewModel.consumeEvent();
                Bundle args = new Bundle();
                args.putLong(ResultFragment.ARG_RECORD_ID, s.recordId);
                NavHostFragment.findNavController(this).navigate(R.id.action_analyze_to_result, args);
            } else if (s.status == AnalyzeState.Status.ERROR) {
                viewModel.consumeEvent();
                Snackbar.make(binding.getRoot(), s.message, Snackbar.LENGTH_LONG)
                        .setAction(R.string.ok, v -> { })
                        .show();
            }
        });
    }

    private void openCamera() {
        try {
            File dir = new File(requireContext().getCacheDir(), "camera");
            if (!dir.exists() && !dir.mkdirs()) throw new IllegalStateException("cache inaccessible");
            File file = new File(dir, "capture_" + System.currentTimeMillis() + ".jpg");
            pendingCameraUri = FileProvider.getUriForFile(requireContext(),
                    requireContext().getPackageName() + ".fileprovider", file);
            takePicture.launch(pendingCameraUri);
        } catch (ActivityNotFoundException | IllegalStateException e) {
            Snackbar.make(binding.getRoot(), R.string.no_camera, Snackbar.LENGTH_LONG).show();
        }
    }

    @Override
    public void onSaveInstanceState(@NonNull Bundle outState) {
        super.onSaveInstanceState(outState);
        if (pendingCameraUri != null) outState.putString("camera_uri", pendingCameraUri.toString());
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
