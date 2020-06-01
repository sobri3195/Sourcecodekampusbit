package com.bagicode.myapplicationtensorflowjava.entity;

import android.graphics.Bitmap;

import com.bagicode.myapplicationtensorflowjava.entity.Classifier;

import java.util.List;

public interface Recognition {

    List<Classifier> recognize(Bitmap bitmap);

    void close();

}
