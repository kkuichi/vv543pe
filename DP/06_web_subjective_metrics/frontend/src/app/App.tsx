import React, { useState, useEffect } from 'react';
import Group1 from './pages/Group1';
import Group2 from './pages/Group2';
import { api } from './services/api';
import { methods } from './data/methods';

interface StudentData {
  student_id: number;
  group_name: 'group1' | 'group2';
  group_title: string;
}

function App() {
  const [student, setStudent] = useState<StudentData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    registerNewStudent();
  }, []);

  const registerNewStudent = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await api.registerStudent();
      console.log("REGISTER RESPONSE:", response);
      
      if (response.success && response.data?.student_id) {
        setStudent(response.data);
      } else {
        setError('student registration error');
      }
    } catch (err) {
      console.error("REGISTER ERROR:", err);
      setError('server connection error');
    } finally {
      setLoading(false);
    }
  };

  const handleComplete = async (data: any) => {
    if (!student) return;

    try {
      setLoading(true);

      console.log("SENDING TO BACK:", {
        student_id: student.student_id,
        ...data
      });

      const response = await api.submitResponse({
        student_id: student.student_id,
        group_name: student.group_name,
        ...data
      });

    } catch (err) {
      console.error("SUBMIT ERROR:", err);
      setError('submit err');
    } finally {
      setLoading(false);
    }
  };

  if (loading && !student) {
    return <div>load...</div>;
  }

  if (error) {
    return (
      <div>
        <p>err: {error}</p>
        <button onClick={registerNewStudent}>try again</button>
      </div>
    );
  }

  if (!student) {
    return <div>no data</div>;
  }

  return (
    <div>
      {student.group_name === 'group1' ? (
        <Group1
          studentId={student.student_id}
          onComplete={handleComplete}
        />
      ) : (
        <Group2
          studentId={student.student_id}
          onComplete={handleComplete}
        />
      )}

      {loading && <div>save...</div>}
    </div>
  );
}

export default App;











