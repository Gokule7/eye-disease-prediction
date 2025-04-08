import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:http/http.dart' as http;

void main() => runApp(EyeDiseaseApp());

class EyeDiseaseApp extends StatelessWidget {
  const EyeDiseaseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Eye Disease Predictor',
      home: UploadPage(),
    );
  }
}

class UploadPage extends StatefulWidget {
  const UploadPage({super.key});

  @override
  _UploadPageState createState() => _UploadPageState();
}

class _UploadPageState extends State<UploadPage> {
  File? leftImage;
  File? rightImage;
  final nameController = TextEditingController();
  final emailController = TextEditingController();

  Future<void> pickImage(bool isLeft) async {
    final pickedFile = await ImagePicker().pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        if (isLeft) {
          leftImage = File(pickedFile.path);
        } else {
          rightImage = File(pickedFile.path);
        }
      });
    }
  }

  Future<void> uploadImages() async {
    if (leftImage == null || rightImage == null || nameController.text.isEmpty || emailController.text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Please provide all inputs")));
      return;
    }

    var uri = Uri.parse("http://172.20.144.152:5000/predict");


    var request = http.MultipartRequest('POST', uri)
      ..fields['name'] = nameController.text
      ..fields['email'] = emailController.text
      ..files.add(await http.MultipartFile.fromPath('left_eye', leftImage!.path))
      ..files.add(await http.MultipartFile.fromPath('right_eye', rightImage!.path));

    var response = await request.send();
    final resBody = await response.stream.bytesToString();
    final decoded = jsonDecode(resBody);

    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: Text("Prediction Result"),
        content: Text(
          "Left Eye: ${decoded['left_eye_result']}\n"
          "Right Eye: ${decoded['right_eye_result']}\n"
          "Status: ${decoded['status']}",
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Eye Disease Predictor")),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: ListView(children: [
          TextField(controller: nameController, decoration: InputDecoration(labelText: "Your Name")),
          TextField(controller: emailController, decoration: InputDecoration(labelText: "Your Email")),
          SizedBox(height: 20),
          ElevatedButton(onPressed: () => pickImage(true), child: Text("Pick Left Eye Image")),
          ElevatedButton(onPressed: () => pickImage(false), child: Text("Pick Right Eye Image")),
          SizedBox(height: 20),
          ElevatedButton(onPressed: uploadImages, child: Text("Submit & Predict")),
        ]),
      ),
    );
  }
}
