<?php

require_once __DIR__ . '/Database.php';

class Response {

    private $pdo;

    public function __construct() {
        $this->pdo = Database::getConnection();
    }

    /**
     * save answer
     * 
     * @param int $student_id 
     * @param string $method 
     * @param array|object $ratings 
     * @return bool - true (200)
     */



     public function save($student_id, $explanations, $rankings = null, $created_at = null) {
        try {
            if (empty($student_id) || empty($explanations)) {
                error_log("Missing required fields");
                return false;
            }
    
            //get group
            $stmtGroup = $this->pdo->prepare(
                "SELECT group_name FROM students WHERE id = ?"
            );
            $stmtGroup->execute([$student_id]);
            $student = $stmtGroup->fetch(PDO::FETCH_ASSOC);
    
            if (!$student) {
                error_log("Student not found");
                return false;
            }
    
            $group_name = $student['group_name'];
    
            //check answers
            $check = $this->pdo->prepare(
                "SELECT id FROM responses WHERE student_id = ?"
            );
            $check->execute([$student_id]);
    
            if ($check->fetch()) {
                return true;
            }
    
            //prep for saving
            $responseData = [
                'explanations' => $explanations,
                'rankings' => $rankings
            ];
    
            //save
            $stmt = $this->pdo->prepare(
                "INSERT INTO responses 
                 (student_id, group_name, response_data, created_at)
                 VALUES (?, ?, ?, ?)"
            );
    
            return $stmt->execute([
                $student_id,
                $group_name,
                // $method,
                json_encode($responseData),
                $created_at ?? date('Y-m-d H:i:s')
            ]);
    
        } catch (PDOException $e) {
            if ($e->getCode() == 23000) {
                return true;
            }
            error_log("Error saving response: " . $e->getMessage());
            return false;
        }
    }


    public function getAll() {
        $stmt = $this->pdo->query(
            "SELECT * FROM responses ORDER BY created_at DESC"
        );
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    }
}



