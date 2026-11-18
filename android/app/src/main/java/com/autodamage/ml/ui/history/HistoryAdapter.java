package com.autodamage.ml.ui.history;

import android.text.format.DateFormat;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.ListAdapter;
import androidx.recyclerview.widget.RecyclerView;

import com.autodamage.ml.R;
import com.autodamage.ml.data.db.PredictionEntity;
import com.autodamage.ml.databinding.ItemHistoryBinding;
import com.autodamage.ml.util.ImageUtils;

import java.util.Date;
import java.util.Locale;

public class HistoryAdapter extends ListAdapter<PredictionEntity, HistoryAdapter.Holder> {

    public interface OnItemClick {
        void onClick(PredictionEntity item);
    }

    private final OnItemClick onClick;

    public HistoryAdapter(OnItemClick onClick) {
        super(DIFF);
        this.onClick = onClick;
    }

    @NonNull
    @Override
    public Holder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        return new Holder(ItemHistoryBinding.inflate(LayoutInflater.from(parent.getContext()), parent, false));
    }

    @Override
    public void onBindViewHolder(@NonNull Holder h, int position) {
        PredictionEntity item = getItem(position);
        Date date = new Date(item.createdAt);
        h.b.imageThumb.setImageBitmap(ImageUtils.loadThumbnail(item.thumbnailPath));
        h.b.textLabel.setText(item.labelFr);
        h.b.textConfidence.setText(String.format(Locale.FRANCE, "%d %%", Math.round(item.confidence * 100)));
        h.b.textDate.setText(h.itemView.getContext().getString(R.string.history_date,
                DateFormat.getMediumDateFormat(h.itemView.getContext()).format(date),
                DateFormat.getTimeFormat(h.itemView.getContext()).format(date)));
        h.b.chipUncertain.setVisibility(item.uncertain ? View.VISIBLE : View.GONE);
        h.itemView.setOnClickListener(v -> onClick.onClick(item));
        h.itemView.setContentDescription(h.itemView.getContext().getString(R.string.history_item_description,
                item.labelFr, Math.round(item.confidence * 100)));
    }

    static class Holder extends RecyclerView.ViewHolder {
        final ItemHistoryBinding b;

        Holder(ItemHistoryBinding b) {
            super(b.getRoot());
            this.b = b;
        }
    }

    private static final DiffUtil.ItemCallback<PredictionEntity> DIFF = new DiffUtil.ItemCallback<PredictionEntity>() {
        @Override
        public boolean areItemsTheSame(@NonNull PredictionEntity a, @NonNull PredictionEntity b) {
            return a.id == b.id;
        }

        @Override
        public boolean areContentsTheSame(@NonNull PredictionEntity a, @NonNull PredictionEntity b) {
            return a.createdAt == b.createdAt && a.label.equals(b.label) && a.confidence == b.confidence;
        }
    };
}
