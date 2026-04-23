<?php
return [
    'db' => [
        'host' => 'localhost',
        'dbname' => 'feedback',
        'user' => 'your_user',
        'pass' => 'your_pass'
    ],
    'groups' => [
        'group1' => [
            'name' => 'Group 1 - BERT',
            'methods' => [
                'LIME',
                'SHAP',
                'IG'
            ]
        ],
        'group2' => [
            'name' => 'Group 2 - HateCLIPper',
            'methods' => [
                'Attention',
                'Occlusion',
                'Gradcam' 
            ]
        ]
    ]
];
?>