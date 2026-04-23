<?php

require_once __DIR__ . '/Database.php';

class Student {

    private $pdo;
    private $config;

    public function __construct() {
        $this->pdo = Database::getConnection();
        $this->config = require __DIR__ . '/../config/config.php';

        if (session_status() === PHP_SESSION_NONE) {
            session_start();
        }
    }

    //new student creation + group balance
    private function registerBalanced() {
        $this->pdo->beginTransaction();

        //amont of srudents in every group
        $stmt = $this->pdo->query(
            "SELECT group_name, COUNT(*) as cnt FROM students GROUP BY group_name FOR UPDATE"
        );

        $counts = [
            'group1' => 0,
            'group2' => 0
        ];

        while ($row = $stmt->fetch(PDO::FETCH_ASSOC)) {
            $counts[$row['group_name']] = $row['cnt'];
        }

        //get group with small amount
        $group_name = ($counts['group1'] <= $counts['group2']) ? 'group1' : 'group2';

        //save student
        $stmt = $this->pdo->prepare(
            "INSERT INTO students (group_name, created_at) 
             VALUES (?,  NOW())"
        );

        $stmt->execute([
            $group_name,
            // json_encode($methods)
        ]);
        $student_id = $this->pdo->lastInsertId();

        $this->pdo->commit();

        return [
            "student_id" => $student_id,
            // "group_id" => $group_name,
            "group_name" => $group_name,
            "group_title" => $this->config['groups'][$group_name]['name'],
            // "method_order" => $methods 
        ];
    }

    //get student by ID
    public function getStudent($student_id) {
        $stmt = $this->pdo->prepare("SELECT * FROM students WHERE id = ?");
        $stmt->execute([$student_id]);
        return $stmt->fetch(PDO::FETCH_ASSOC);
    }

    // get or create student
    public function getOrRegister() {

    // if exists
    if (!empty($_SESSION['student_id'])) {
        $student = $this->getStudent($_SESSION['student_id']);
        if ($student) {
            return [
                "student_id" => $student["id"],
                "group_name" => $student["group_name"],
                // "method_order" => json_decode($student["method_order"], true),
                "created_at" => $student["created_at"]
            ];
        }
    }

    // if in cookie
    if (!empty($_COOKIE['student_id'])) {
        $student = $this->getStudent($_COOKIE['student_id']);
        if ($student) {
            $_SESSION['student_id'] = $student["id"];
            return [
                "student_id" => $student["id"],
                "group_name" => $student["group_name"],
                // "method_order" => json_decode($student["method_order"], true),
                "created_at" => $student["created_at"]
            ];
        }
    }

    // create new
    $student = $this->registerBalanced();

    $_SESSION['student_id'] = $student["student_id"];
    setcookie('student_id', $student["student_id"], time() + 3600*24*30, "/", "", false, true);
    return $student;
}
}




