import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:dio/dio.dart';

class EyeDiseaseUploadScreen extends StatefulWidget {
  const EyeDiseaseUploadScreen({super.key});

  @override
  _EyeDiseaseUploadScreenState createState() => _EyeDiseaseUploadScreenState();
}

class _EyeDiseaseUploadScreenState extends State<EyeDiseaseUploadScreen> {
  File? _leftEyeImage;
  File? _rightEyeImage;
  String _predictionResult = '';
  bool _isLoading = false;
  final _emailController = TextEditingController();

  final picker = ImagePicker();

  Future pickImage(bool isLeft) async {
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        if (isLeft) {
          _leftEyeImage = File(pickedFile.path);
        } else {
          _rightEyeImage = File(pickedFile.path);
        }
      });
    }
  }

  Future<void> uploadAndPredict() async {
    if (_leftEyeImage == null || _rightEyeImage == null) {
      setState(() {
        _predictionResult = 'Please select both left and right eye images.';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _predictionResult = '';
    });

    var dio = Dio();
    var formData = FormData.fromMap({
      "left_eye": await MultipartFile.fromFile(_leftEyeImage!.path),
      "right_eye": await MultipartFile.fromFile(_rightEyeImage!.path),
    });

    try {
      final response = await dio.post(
        "http://192.168.230.227:5000/predict"
, // Replace with your IP address
        data: formData,
      );

      setState(() {
        _predictionResult = response.data.toString();
        _isLoading = false;
      });

      // Optionally send result to email
      if (_emailController.text.isNotEmpty) {
        await dio.post("http://192.168.230.227:5000/send_email"
, data: {
          "email": _emailController.text,
          "result": response.data.toString(),
        });
      }
    } catch (e) {
      setState(() {
        _predictionResult = 'Prediction failed. Error: \$e';
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Eye Disease Prediction')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            ElevatedButton(
              onPressed: () => pickImage(true),
              child: Text('Pick Left Eye Image'),
            ),
            _leftEyeImage != null
                ? Image.file(_leftEyeImage!, height: 100)
                : SizedBox(),
            SizedBox(height: 16),
            ElevatedButton(
              onPressed: () => pickImage(false),
              child: Text('Pick Right Eye Image'),
            ),
            _rightEyeImage != null
                ? Image.file(_rightEyeImage!, height: 100)
                : SizedBox(),
            SizedBox(height: 16),
            TextField(
              controller: _emailController,
              decoration: InputDecoration(
                labelText: 'Enter Email (Optional)',
                border: OutlineInputBorder(),
              ),
            ),
            SizedBox(height: 16),
            ElevatedButton(
              onPressed: _isLoading ? null : uploadAndPredict,
              child: _isLoading
                  ? CircularProgressIndicator(color: Colors.white)
                  : Text('Upload & Predict'),
            ),
            SizedBox(height: 16),
            Text(
              _predictionResult,
              style: TextStyle(fontSize: 14, color: Colors.black87),
            )
          ],
        ),
      ),
    );
  }
}
