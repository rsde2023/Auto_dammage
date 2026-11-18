package com.autodamage.ml.ui.analyze;

/** État de l'écran d'analyse, observé par le fragment. */
public final class AnalyzeState {
    public enum Status { IDLE, LOADING_PHOTO, ANALYZING, SUCCESS, ERROR }

    public final Status status;
    public final String message;   // message d'erreur (ERROR)
    public final long recordId;    // enregistrement créé (SUCCESS)

    private AnalyzeState(Status status, String message, long recordId) {
        this.status = status;
        this.message = message;
        this.recordId = recordId;
    }

    public static AnalyzeState idle() { return new AnalyzeState(Status.IDLE, null, -1); }
    public static AnalyzeState loadingPhoto() { return new AnalyzeState(Status.LOADING_PHOTO, null, -1); }
    public static AnalyzeState analyzing() { return new AnalyzeState(Status.ANALYZING, null, -1); }
    public static AnalyzeState success(long id) { return new AnalyzeState(Status.SUCCESS, null, id); }
    public static AnalyzeState error(String message) { return new AnalyzeState(Status.ERROR, message, -1); }

    public boolean isBusy() {
        return status == Status.LOADING_PHOTO || status == Status.ANALYZING;
    }
}
