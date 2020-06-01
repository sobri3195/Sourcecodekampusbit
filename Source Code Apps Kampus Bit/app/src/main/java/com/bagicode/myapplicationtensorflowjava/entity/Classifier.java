package com.bagicode.myapplicationtensorflowjava.entity;

public class Classifier {

    private String id;
    private String title;
    private Float confidence;

    public Classifier(String id, String title, Float confidence) {
        this.id = id;
        this.title = title;
        this.confidence = confidence;
    }

    public String getId() {
        return id;
    }

    public String getTitle() {
        return title;
    }

    public Float getConfidence() {
        return confidence;
    }
}
