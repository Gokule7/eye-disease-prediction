import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:url_launcher/url_launcher.dart';
import 'package:image_picker/image_picker.dart';

class ImageUploadScreen extends StatefulWidget {
  const ImageUploadScreen({super.key});

  @override
  _ImageUploadScreenState createState() => _ImageUploadScreenState();
}

class _ImageUploadScreenState extends State<ImageUploadScreen> {
  File? leftEye;
  File? rightEye;

  final picker = ImagePicker();
  final TextEditingController emailController = TextEditingController();
  final TextEditingController nameController = TextEditingController();

  Future<void> pickImage(bool isLeft) async {
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        if (isLeft) {
          leftEye = File(pickedFile.path);
        } else {
          rightEye = File(pickedFile.path);
        }
      });
    }
  }

  Future<void> uploadAndGetPDF() async {
    if (leftEye == null || rightEye == null) return;

    var uri = Uri.parse("http://172.29.192.1:5000/predict");

    var request = http.MultipartRequest('POST', uri);

    request.files.add(await http.MultipartFile.fromPath('left_eye', leftEye!.path));
    request.files.add(await http.MultipartFile.fromPath('right_eye', rightEye!.path));
    request.fields['email'] = emailController.text.trim();
    request.fields['name'] = nameController.text.trim();

    var response = await request.send();

    if (response.statusCode == 200) {
      var data = json.decode(await response.stream.bytesToString());
      String pdfUrl = data['pdf_url'];

      if (await canLaunchUrl(Uri.parse(pdfUrl))) {
        await launchUrl(Uri.parse(pdfUrl), mode: LaunchMode.externalApplication);
      } else {
        throw 'Could not open PDF';
      }
    } else {
      print("Upload failed: ${response.statusCode}");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Eye Report Upload")),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(controller: nameController, decoration: InputDecoration(labelText: "Name")),
            TextField(controller: emailController, decoration: InputDecoration(labelText: "Email")),
            ElevatedButton(onPressed: () => pickImage(true), child: Text("Pick Left Eye")),
            ElevatedButton(onPressed: () => pickImage(false), child: Text("Pick Right Eye")),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: uploadAndGetPDF,
              child: Text("Upload and Get Report"),
            ),
          ],
        ),
      ),
    );
  }
}
