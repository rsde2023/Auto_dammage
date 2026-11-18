package com.autodamage.ml.ui.widget;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

/**
 * Affiche la photo et un cadre que l'utilisateur déplace (glisser à l'intérieur) et redimensionne (glisser un coin).
 * Le cadre doit entourer la zone endommagée AU PLUS PRÈS : le modèle a été entraîné sur des zones sans marge.
 * Le cadre est exposé en coordonnées normalisées (0..1), indépendantes de la taille de l'écran.
 */
public class CropOverlayView extends View {

    public interface OnCropChangeListener {
        void onCropChanged(RectF normalized);
    }

    private static final int NONE = 0, MOVE = 1, TOP_LEFT = 2, TOP_RIGHT = 3, BOTTOM_LEFT = 4, BOTTOM_RIGHT = 5;

    private final Paint bitmapPaint = new Paint(Paint.FILTER_BITMAP_FLAG);
    private final Paint dimPaint = new Paint();
    private final Paint borderPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint gridPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint handlePaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    private final RectF imageRect = new RectF();   // position de la photo à l'écran
    private final RectF crop = new RectF();        // cadre à l'écran
    private final RectF normalized = new RectF(0.2f, 0.2f, 0.8f, 0.8f);

    private final float density = getResources().getDisplayMetrics().density;
    private final float handleRadius = 9 * density;
    private final float touchRadius = 28 * density;
    private final float minSize = 40 * density;

    private Bitmap bitmap;
    private int mode = NONE;
    private float lastX, lastY;
    private OnCropChangeListener listener;

    public CropOverlayView(Context context) {
        this(context, null);
    }

    public CropOverlayView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        dimPaint.setColor(Color.argb(150, 0, 0, 0));
        borderPaint.setColor(Color.WHITE);
        borderPaint.setStyle(Paint.Style.STROKE);
        borderPaint.setStrokeWidth(2 * density);
        gridPaint.setColor(Color.argb(110, 255, 255, 255));
        gridPaint.setStrokeWidth(1 * density);
        handlePaint.setColor(Color.WHITE);
        setContentDescription("Cadre de sélection de la zone endommagée");
    }

    public void setImageBitmap(@Nullable Bitmap bmp) {
        bitmap = bmp;
        layoutImage();
        invalidate();
    }

    public void setOnCropChangeListener(OnCropChangeListener l) {
        listener = l;
    }

    /** Cadre normalisé (fractions de la largeur et de la hauteur de la photo). */
    public RectF getCropNormalized() {
        return new RectF(normalized);
    }

    public void setCropNormalized(@NonNull RectF n) {
        normalized.set(n);
        applyNormalized();
        invalidate();
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        layoutImage();
    }

    private void layoutImage() {
        if (bitmap == null || getWidth() == 0 || getHeight() == 0) return;
        float vw = getWidth() - getPaddingLeft() - getPaddingRight();
        float vh = getHeight() - getPaddingTop() - getPaddingBottom();
        float scale = Math.min(vw / bitmap.getWidth(), vh / bitmap.getHeight());
        float dw = bitmap.getWidth() * scale, dh = bitmap.getHeight() * scale;
        float left = getPaddingLeft() + (vw - dw) / 2f, top = getPaddingTop() + (vh - dh) / 2f;
        imageRect.set(left, top, left + dw, top + dh);
        applyNormalized();
    }

    private void applyNormalized() {
        if (imageRect.isEmpty()) return;
        crop.set(imageRect.left + normalized.left * imageRect.width(),
                imageRect.top + normalized.top * imageRect.height(),
                imageRect.left + normalized.right * imageRect.width(),
                imageRect.top + normalized.bottom * imageRect.height());
    }

    private void updateNormalized() {
        normalized.set((crop.left - imageRect.left) / imageRect.width(),
                (crop.top - imageRect.top) / imageRect.height(),
                (crop.right - imageRect.left) / imageRect.width(),
                (crop.bottom - imageRect.top) / imageRect.height());
        if (listener != null) listener.onCropChanged(getCropNormalized());
    }

    @Override
    protected void onDraw(@NonNull Canvas canvas) {
        super.onDraw(canvas);
        if (bitmap == null || imageRect.isEmpty()) return;
        canvas.drawBitmap(bitmap, null, imageRect, bitmapPaint);
        // assombrit tout ce qui est hors du cadre
        canvas.drawRect(imageRect.left, imageRect.top, imageRect.right, crop.top, dimPaint);
        canvas.drawRect(imageRect.left, crop.bottom, imageRect.right, imageRect.bottom, dimPaint);
        canvas.drawRect(imageRect.left, crop.top, crop.left, crop.bottom, dimPaint);
        canvas.drawRect(crop.right, crop.top, imageRect.right, crop.bottom, dimPaint);
        // repères des tiers, bordure, poignées
        for (int i = 1; i <= 2; i++) {
            float x = crop.left + crop.width() * i / 3f, y = crop.top + crop.height() * i / 3f;
            canvas.drawLine(x, crop.top, x, crop.bottom, gridPaint);
            canvas.drawLine(crop.left, y, crop.right, y, gridPaint);
        }
        canvas.drawRect(crop, borderPaint);
        canvas.drawCircle(crop.left, crop.top, handleRadius, handlePaint);
        canvas.drawCircle(crop.right, crop.top, handleRadius, handlePaint);
        canvas.drawCircle(crop.left, crop.bottom, handleRadius, handlePaint);
        canvas.drawCircle(crop.right, crop.bottom, handleRadius, handlePaint);
    }

    @Override
    public boolean onTouchEvent(MotionEvent e) {
        if (bitmap == null || imageRect.isEmpty()) return false;
        float x = e.getX(), y = e.getY();
        switch (e.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
                mode = hitTest(x, y);
                lastX = x;
                lastY = y;
                if (mode != NONE) getParent().requestDisallowInterceptTouchEvent(true);
                return mode != NONE;
            case MotionEvent.ACTION_MOVE:
                if (mode == MOVE) {
                    float dx = clampDelta(x - lastX, crop.left - imageRect.left, imageRect.right - crop.right);
                    float dy = clampDelta(y - lastY, crop.top - imageRect.top, imageRect.bottom - crop.bottom);
                    crop.offset(dx, dy);
                } else if (mode != NONE) {
                    resize(x, y);
                }
                lastX = x;
                lastY = y;
                updateNormalized();
                invalidate();
                return true;
            case MotionEvent.ACTION_UP:
                performClick();
                // fall through
            case MotionEvent.ACTION_CANCEL:
                mode = NONE;
                getParent().requestDisallowInterceptTouchEvent(false);
                return true;
            default:
                return false;
        }
    }

    @Override
    public boolean performClick() {
        return super.performClick();
    }

    private int hitTest(float x, float y) {
        if (near(x, y, crop.left, crop.top)) return TOP_LEFT;
        if (near(x, y, crop.right, crop.top)) return TOP_RIGHT;
        if (near(x, y, crop.left, crop.bottom)) return BOTTOM_LEFT;
        if (near(x, y, crop.right, crop.bottom)) return BOTTOM_RIGHT;
        return crop.contains(x, y) ? MOVE : NONE;
    }

    private boolean near(float x, float y, float cx, float cy) {
        return Math.abs(x - cx) <= touchRadius && Math.abs(y - cy) <= touchRadius;
    }

    private void resize(float x, float y) {
        x = Math.max(imageRect.left, Math.min(imageRect.right, x));
        y = Math.max(imageRect.top, Math.min(imageRect.bottom, y));
        float min = Math.min(minSize, Math.min(imageRect.width(), imageRect.height()));
        if (mode == TOP_LEFT || mode == BOTTOM_LEFT) crop.left = Math.min(x, crop.right - min);
        if (mode == TOP_RIGHT || mode == BOTTOM_RIGHT) crop.right = Math.max(x, crop.left + min);
        if (mode == TOP_LEFT || mode == TOP_RIGHT) crop.top = Math.min(y, crop.bottom - min);
        if (mode == BOTTOM_LEFT || mode == BOTTOM_RIGHT) crop.bottom = Math.max(y, crop.top + min);
    }

    private static float clampDelta(float d, float roomBefore, float roomAfter) {
        return Math.max(-roomBefore, Math.min(roomAfter, d));
    }
}
