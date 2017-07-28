package com.cv.hy.opencvinpicture;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    public static final int TAKE_PHOTE = 1;

    private Button takePhote;
    private ImageView picture_src;
    private ImageView picture_detect;
    private Uri imageUri;
    private File imageFile;

    private CascadeClassifier cascadeClassifier;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        takePhote = (Button) findViewById(R.id.take_photo);
        picture_detect = (ImageView) findViewById(R.id.picture_detect);
        picture_src = (ImageView) findViewById(R.id.picture_src);
        takePhote.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                File outputImage = new File(Environment.getExternalStorageDirectory(), "output_image.jpg");
                try {
                    if (outputImage.exists()) {
                        outputImage.delete();
                    }
                    outputImage.createNewFile();
                } catch (IOException e) {
                    e.printStackTrace();
                }
                imageUri = Uri.fromFile(outputImage);
                imageFile = outputImage;
                Intent intent = new Intent("android.media.action.IMAGE_CAPTURE");
                intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
                startActivityForResult(intent, TAKE_PHOTE);//Start camera
            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        OpenCVLoader.initDebug();//一定要这样初始化！！！！
        initializeOpenCVDependencies();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        switch (requestCode) {
            case TAKE_PHOTE:
                if (resultCode == RESULT_OK) {
//                    Bitmap bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(imageUri));
                    Bitmap bitmap = getSmallBitmap(imageFile.getPath());
                    Log.i("Main->bitmap after pro:", bitmap.getWidth() + "  " + bitmap.getHeight());
                    picture_detect.setImageBitmap(detect(bitmap));

                    try {
                        showGray();
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    }
                }
                break;
            default:
                break;
        }
    }

    private void showGray() throws FileNotFoundException {
        Mat rgbMat = new Mat();
        Mat grayMat = new Mat();
//        Bitmap rgbBitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(imageUri));
        Bitmap rgbBitmap = getSmallBitmap(imageFile.getPath());
        Bitmap grayBitmap = Bitmap.createBitmap(rgbBitmap.getWidth(), rgbBitmap.getHeight(), Bitmap.Config.RGB_565);
        Utils.bitmapToMat(rgbBitmap, rgbMat);
        Imgproc.cvtColor(rgbMat, grayMat, Imgproc.COLOR_RGB2GRAY);
        MatOfRect imageDetections = new MatOfRect();
        if (cascadeClassifier != null) {
            cascadeClassifier.detectMultiScale(grayMat, imageDetections, 1.1, 2, 2, new Size(50, 50), new Size());
        }
        for (Rect rect : imageDetections.toArray()) {
            Imgproc.rectangle(grayMat, rect.tl(), rect.br(), new Scalar(255, 255, 255, 255), 10);
        }

        Utils.matToBitmap(grayMat, grayBitmap);
        picture_src.setImageBitmap(grayBitmap);
    }

    private Bitmap detect(Bitmap bitmap) {
        Mat imageMat = new Mat();
        Mat srcMat = new Mat();
        Utils.bitmapToMat(bitmap, srcMat);
        Imgproc.cvtColor(srcMat, imageMat, Imgproc.COLOR_RGBA2RGB);

        MatOfRect imageDetections = new MatOfRect();
        if (cascadeClassifier != null) {
            cascadeClassifier.detectMultiScale(imageMat, imageDetections, 1.1, 2, 0, new Size(50, 50), new Size());
        }
        for (Rect rect : imageDetections.toArray()) {
            Imgproc.rectangle(imageMat, rect.tl(), rect.br(), new Scalar(0, 255, 0, 255), 10);
        }

        Utils.matToBitmap(imageMat, bitmap);
        return bitmap;
    }

    private Bitmap getSmallBitmap(String filePath) {
        final BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(filePath, options);
        // Calculate inSampleSize
        options.inSampleSize = calculateInSampleSize(options, 600, 800);

        // Decode bitmap with inSampleSize set
        options.inJustDecodeBounds = false;

        return BitmapFactory.decodeFile(filePath, options);
    }

    private int calculateInSampleSize(BitmapFactory.Options options, int reqWidth, int reqHeight) {
        final int height = options.outHeight;
        final int width = options.outWidth;
        int inSampleSize = 1;

        if (height > reqHeight || width > reqWidth) {
            final int heightRatio = Math.round((float) height / (float) reqHeight);
            final int widthRatio = Math.round((float) width / (float) reqWidth);
            inSampleSize = heightRatio < widthRatio ? heightRatio : widthRatio;
        }
        return inSampleSize;
    }


    private void initializeOpenCVDependencies() {
        try {
            // Copy the resource into a temp file so OpenCV can load it
//            InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
            InputStream is = getResources().openRawResource(R.raw.cascade);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
//            File mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            File mCascadeFile = new File(cascadeDir, "cascade.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);


            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            // Load the cascade classifier
            cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
        } catch (Exception e) {
            Log.e("OpenCVActivity", "Error loading cascade", e);
        }
    }
}
