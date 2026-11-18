package com.autodamage.ml.util;

import android.content.ContentResolver;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;

import androidx.exifinterface.media.ExifInterface;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * Opérations sur les photos côté téléphone : lecture, rotation EXIF, réduction, découpe, JPEG.
 * Aucun calcul de Machine Learning ici : le prétraitement du modèle est fait par le serveur.
 */
public final class ImageUtils {
    /** Plus grand côté de la photo envoyée : proche de la taille des photos CarDD (~1000 px). */
    public static final int MAX_UPLOAD_SIDE = 1280;
    public static final int THUMBNAIL_SIDE = 256;

    private ImageUtils() { }

    /** Lit la photo, la remet à l'endroit (EXIF) et la réduit à maxSide pixels au plus. */
    public static Bitmap loadUpright(ContentResolver resolver, Uri uri, int maxSide) throws IOException {
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        try (InputStream in = resolver.openInputStream(uri)) {
            BitmapFactory.decodeStream(in, null, bounds);
        }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            throw new IOException("Fichier image illisible");
        }
        BitmapFactory.Options opts = new BitmapFactory.Options();
        opts.inSampleSize = 1;
        while (Math.max(bounds.outWidth, bounds.outHeight) / (opts.inSampleSize * 2) >= maxSide) {
            opts.inSampleSize *= 2;          // décodage déjà réduit : économise la mémoire
        }
        Bitmap bitmap;
        try (InputStream in = resolver.openInputStream(uri)) {
            bitmap = BitmapFactory.decodeStream(in, null, opts);
        }
        if (bitmap == null) throw new IOException("Fichier image illisible");

        int orientation;
        try (InputStream in = resolver.openInputStream(uri)) {
            orientation = in == null ? ExifInterface.ORIENTATION_NORMAL
                    : new ExifInterface(in).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
        }
        return scaleDown(rotate(bitmap, orientation), maxSide);
    }

    static Bitmap rotate(Bitmap src, int orientation) {
        Matrix m = new Matrix();
        switch (orientation) {
            case ExifInterface.ORIENTATION_ROTATE_90: m.postRotate(90); break;
            case ExifInterface.ORIENTATION_ROTATE_180: m.postRotate(180); break;
            case ExifInterface.ORIENTATION_ROTATE_270: m.postRotate(270); break;
            case ExifInterface.ORIENTATION_FLIP_HORIZONTAL: m.postScale(-1, 1); break;
            case ExifInterface.ORIENTATION_FLIP_VERTICAL: m.postScale(1, -1); break;
            default: return src;
        }
        Bitmap out = Bitmap.createBitmap(src, 0, 0, src.getWidth(), src.getHeight(), m, true);
        if (out != src) src.recycle();
        return out;
    }

    public static Bitmap scaleDown(Bitmap src, int maxSide) {
        int w = src.getWidth(), h = src.getHeight();
        if (Math.max(w, h) <= maxSide) return src;
        float s = (float) maxSide / Math.max(w, h);
        return Bitmap.createScaledBitmap(src, Math.max(1, Math.round(w * s)), Math.max(1, Math.round(h * s)), true);
    }

    public static Bitmap crop(Bitmap src, int[] bbox) {
        return Bitmap.createBitmap(src, bbox[0], bbox[1], bbox[2], bbox[3]);
    }

    public static byte[] toJpeg(Bitmap bitmap, int quality) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, quality, out);
        return out.toByteArray();
    }

    public static void saveJpeg(Bitmap bitmap, File file, int quality) throws IOException {
        File parent = file.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("Impossible de créer " + parent);
        }
        try (FileOutputStream out = new FileOutputStream(file)) {
            bitmap.compress(Bitmap.CompressFormat.JPEG, quality, out);
        }
    }

    /** Vignette de l'historique : petite, décodée rapidement. */
    public static Bitmap loadThumbnail(String path) {
        return path == null ? null : BitmapFactory.decodeFile(path);
    }
}
