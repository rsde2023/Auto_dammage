package com.autodamage.ml.ui.history;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.view.MenuProvider;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.Lifecycle;
import androidx.lifecycle.ViewModelProvider;
import androidx.navigation.fragment.NavHostFragment;

import com.autodamage.ml.R;
import com.autodamage.ml.databinding.FragmentHistoryBinding;
import com.autodamage.ml.ui.result.ResultFragment;
import com.google.android.material.dialog.MaterialAlertDialogBuilder;

/** Écran 3 : historique local des analyses (Room). */
public class HistoryFragment extends Fragment {

    private FragmentHistoryBinding binding;
    private HistoryViewModel viewModel;
    private boolean hasItems;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle state) {
        binding = FragmentHistoryBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        viewModel = new ViewModelProvider(this).get(HistoryViewModel.class);
        HistoryAdapter adapter = new HistoryAdapter(item -> {
            Bundle args = new Bundle();
            args.putLong(ResultFragment.ARG_RECORD_ID, item.id);
            NavHostFragment.findNavController(this).navigate(R.id.action_history_to_result, args);
        });
        binding.recycler.setAdapter(adapter);
        viewModel.history().observe(getViewLifecycleOwner(), items -> {
            adapter.submitList(items);
            hasItems = items != null && !items.isEmpty();
            binding.emptyState.setVisibility(hasItems ? View.GONE : View.VISIBLE);
            requireActivity().invalidateOptionsMenu();
        });

        requireActivity().addMenuProvider(new MenuProvider() {
            @Override
            public void onCreateMenu(@NonNull Menu menu, @NonNull MenuInflater inflater) {
                inflater.inflate(R.menu.menu_history, menu);
            }

            @Override
            public void onPrepareMenu(@NonNull Menu menu) {
                MenuItem clear = menu.findItem(R.id.action_clear_history);
                if (clear != null) clear.setVisible(hasItems);
            }

            @Override
            public boolean onMenuItemSelected(@NonNull MenuItem item) {
                if (item.getItemId() == R.id.action_clear_history) {
                    new MaterialAlertDialogBuilder(requireContext())
                            .setTitle(R.string.clear_history_title)
                            .setMessage(R.string.clear_history_message)
                            .setNegativeButton(R.string.cancel, null)
                            .setPositiveButton(R.string.delete, (d, w) -> viewModel.clear())
                            .show();
                    return true;
                }
                return false;
            }
        }, getViewLifecycleOwner(), Lifecycle.State.RESUMED);
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
