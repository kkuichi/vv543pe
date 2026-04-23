<?php


$allowed_origin = "";
header("Access-Control-Allow-Origin: $allowed_origin");
header("Access-Control-Allow-Credentials: true"); 
header("Access-Control-Allow-Methods: GET, POST, OPTIONS");
header("Access-Control-Allow-Headers: Content-Type");
header("Content-Type: application/json");
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit();
}

if (session_status() === PHP_SESSION_NONE) {
    session_start();
}

require_once __DIR__ . '/../src/Student.php';
require_once __DIR__ . '/../src/Response.php';

try {
    $method = $_SERVER['REQUEST_METHOD'];

    if ($method === "GET") {
        // GET
        $studentModel = new Student();
        $result = $studentModel->getOrRegister();

        // save student_id 
        // $_SESSION['student_id'] = $result['student_id'];

        echo json_encode([
            "success" => true,
            "data" => $result
        ]);
        exit();

    } elseif ($method === "POST") {
        // POST
        $data = json_decode(file_get_contents("php://input"), true);

        if (!$data) {
            throw new Exception("Invalid JSON data");
        }

        //check answers
        if (!isset($data['explanations'])) {
            throw new Exception("Missing required fields: method and explanations");
        }

        // get student_id,  POST 
        $student_id = $data['student_id'] ?? $_SESSION['student_id'] ?? null;
        if (!$student_id) {
            throw new Exception("Student ID not found. Reload the page to register.");
        }

        //save to db
        $response = new Response();
        $saved = $response->save(
            $student_id,
            // $data['method'],
            json_encode($data['explanations']),
            isset($data['rankings']) ? json_encode($data['rankings']) : null,
            date('Y-m-d H:i:s')
        );

        if (!$saved) {
            throw new Exception("Failed to save response to the database");
        }

        echo json_encode([
            "success" => true,
            "status" => "saved",
            "message" => "Response saved successfully"
        ]);
        exit();

    } else {
        http_response_code(405);
        echo json_encode([
            "success" => false,
            "error" => "Method not allowed"
        ]);
        exit();
    }

} catch (Exception $e) {
    http_response_code(500);
    echo json_encode([
        "success" => false,
        "error" => $e->getMessage()
    ]);
    exit();
}
 
