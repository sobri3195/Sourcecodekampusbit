package com.bagicode.myapplicationtensorflowjava;

import android.graphics.Bitmap;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Toast;

import com.bagicode.myapplicationtensorflowjava.core.TensorflowClassifier;
import com.bagicode.myapplicationtensorflowjava.entity.Classifier;
import com.bagicode.myapplicationtensorflowjava.entity.Recognition;
import com.bumptech.glide.Glide;
import com.bumptech.glide.request.animation.GlideAnimation;
import com.bumptech.glide.request.target.SimpleTarget;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private ImageView imgObject;
    private EditText edtUrl;
    private Button btnPredict;

    private static String MODEL = "mobilenet_quant_v1_224.tflite";
    private static String LABEL = "labels.txt";
    private static int IMG_SIZE = 224;

    private Recognition recognition;
    private Executor executor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imgObject = findViewById(R.id.img_object);
        edtUrl = findViewById(R.id.edt_url);
        btnPredict = findViewById(R.id.btn_predict);

        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    recognition = TensorflowClassifier.init(
                            getAssets(),
                            MODEL,
                            LABEL,
                            IMG_SIZE
                    );
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });

        btnPredict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String url = edtUrl.getText().toString();
                Glide.with(MainActivity.this)
                        .load(url)
                        .asBitmap()
                        .into(new SimpleTarget<Bitmap>() {
                            @Override
                            public void onResourceReady(Bitmap resource, GlideAnimation<? super Bitmap> glideAnimation) {
                                Bitmap bitmap = Bitmap.createScaledBitmap(
                                        resource,
                                        IMG_SIZE,
                                        IMG_SIZE,
                                        false
                                );

                                imgObject.setImageBitmap(bitmap);

                                String label = "";
                                List<Classifier> resultPredict = recognition.recognize(bitmap);
                                for (Classifier result: resultPredict) {
                                    label+= result.getTitle() + " ( "+ result.getConfidence() + " ) " + "\n";
                                }

                                Toast.makeText(MainActivity.this, label, Toast.LENGTH_SHORT).show();
                                Log.v("Tamvan", label);
                            }
                        });
            }
        });


    }
}
