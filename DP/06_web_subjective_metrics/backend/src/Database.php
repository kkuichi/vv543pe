<?php

class Database {
    private static $pdo = null;
    
    public static function getConnection() {
        if (self::$pdo === null) {
            $config = require __DIR__ . '/../config/config.php';
            $dbConfig = $config['db']; 
            
            $dsn = "mysql:host={$dbConfig['host']};dbname={$dbConfig['dbname']};charset=utf8mb4";
            
            self::$pdo = new PDO(
                $dsn,
                $dbConfig['user'],
                $dbConfig['pass']
            );
            
            self::$pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        }
        
        return self::$pdo;
    }
}