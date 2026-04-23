const API_URL = '';

export const api = {
  async registerStudent() {
    try {
      const response = await fetch(API_URL, {
        method: 'GET',
        credentials: 'include'
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'registration err');
      }
      
      return data; // { success: true, data: { student_id, group_name, method_order } }
      
    } catch (error) {
      console.error('Register student error:', error);
      throw error;
    }
  },


  async submitResponse(payload: {
    student_id: number;
    group_name: string;
    explanations: any;
    rankings: any;
    completed_at: string;
  }) {
    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          action: 'submit_response',  
          ...payload
        })
      });
  
      const data = await response.json();
  
      if (!response.ok) {
        throw new Error(data.error || 'submit err');
      }
  
      return data;
  
    } catch (error) {
      console.error('Submit response error:', error);
      throw error;
    }
  },


};