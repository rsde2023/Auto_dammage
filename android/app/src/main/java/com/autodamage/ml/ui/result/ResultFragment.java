package com.autodamage.ml.ui.result;

import android.os.Bundle;
import android.text.format.DateFormat;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.navigation.NavController;
import androidx.navigation.fragment.NavHostFragment;

import com.autodamage.ml.R;
import com.autodamage.ml.data.api.ClassScore;
import com.autodamage.ml.data.db.PredictionEntity;
import com.autodamage.ml.databinding.FragmentResultBinding;
import com.autodamage.ml.databinding.ItemScoreBinding;
import com.autodamage.ml.util.ImageUtils;
import com.autodamage.ml.util.ScoresJson;

import java.util.Date;
import java.util.Locale;

/** Écran 2 : résultat d'une analyse (nouvelle ou ouverte depuis l'historique). */
public class ResultFragment extends Fragment {
    public static final String ARG_RECORD_ID = "recordId";

    private FragmentResultBinding binding;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle state) {
        binding = FragmentResultBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        long id = requireArguments().getLong(ARG_RECORD_ID, -1);
        ResultViewModel vm = new ViewModelProvider(this).get(ResultViewModel.class);
        vm.record(id).observe(getViewLifecycleOwner(), this::show);
        binding.buttonNewAnalysis.setOnClickListener(v -> {
            // Retour à l'écran d'analyse (la photo reste chargée : on peut encadrer une autre zone).
            NavController nav = NavHostFragment.findNavController(this);
            if (!nav.popBackStack(R.id.analyzeFragment, false)) nav.navigate(R.id.analyzeFragment);
        });
    }

    private void show(@Nullable PredictionEntity r) {
        if (r == null) return;
        binding.imageRegion.setImageBitmap(ImageUtils.loadThumbnail(r.thumbnailPath));
        binding.textLabel.setText(r.labelFr);
        binding.textConfidence.setText(getString(R.string.confidence_value, percent(r.confidence)));
        binding.cardUncertain.setVisibility(r.uncertain ? View.VISIBLE : View.GONE);
        binding.textMeta.setText(getString(R.string.result_meta,
                DateFormat.getMediumDateFormat(requireContext()).format(new Date(r.createdAt)) + " "
                        + DateFormat.getTimeFormat(requireContext()).format(new Date(r.createdAt)),
                r.model == null ? "?" : r.model.toUpperCase(Locale.ROOT)));
        binding.textDisclaimer.setText(r.disclaimer != null ? r.disclaimer : getString(R.string.disclaimer_default));

        binding.scoresContainer.removeAllViews();
        for (ClassScore s : ScoresJson.fromJson(r.topJson)) {
            ItemScoreBinding row = ItemScoreBinding.inflate(getLayoutInflater(), binding.scoresContainer, true);
            row.textScoreLabel.setText(s.labelFr);
            row.textScoreValue.setText(percent(s.probability));
            row.progressScore.setProgress((int) Math.round(s.probability * 100));
        }
    }

    private static String percent(double p) {
        return String.format(Locale.FRANCE, "%d %%", Math.round(p * 100));
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
