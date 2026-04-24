import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';

const ForgotPassword = () => {
  const [email, setEmail] = useState('');
  const [otp, setOtp] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [phase, setPhase] = useState(1); // 1: Email, 2: OTP/Reset
  const [message, setMessage] = useState({ text: '', isError: false });
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const API_BASE_URL = "http://localhost:8000/api/auth";

  // Phase 1: Request OTP
  const handleRequestOTP = async (e) => {
    e.preventDefault();
    setLoading(true);
    setMessage({ text: '', isError: false });

    try {
      const response = await fetch(`${API_BASE_URL}/forgot-password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email }),
      });

      const data = await response.json();

      if (response.ok) {
        setPhase(2);
        setMessage({ text: 'OTP sent to your email!', isError: false });
      } else {
        setMessage({ text: data.detail || 'Failed to send OTP', isError: true });
      }
    } catch (error) {
      setMessage({ text: 'Server error. Please try again later.', isError: true });
    } finally {
      setLoading(false);
    }
  };

  // Phase 2: Reset Password
  const handleResetPassword = async (e) => {
    e.preventDefault();
    if (newPassword.length < 8) {
      setMessage({ text: 'Password must be at least 8 characters.', isError: true });
      return;
    }
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/reset-password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, otp, new_password: newPassword }),
      });

      const data = await response.json();

      if (response.ok) {
        setMessage({ text: 'Password reset successful! Redirecting...', isError: false });
        setTimeout(() => navigate('/login'), 2500);
      } else {
        setMessage({ text: data.detail || 'Invalid or expired OTP', isError: true });
      }
    } catch (error) {
      setMessage({ text: 'Server error. Please try again later.', isError: true });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100 p-4">
      <div className="w-full max-w-md bg-white rounded-xl shadow-lg p-8">
        <h2 className="text-2xl font-bold text-center text-blue-700 mb-6">
          Password Recovery
        </h2>

        {message.text && (
          <div className={`p-3 rounded mb-4 text-sm text-center ${
            message.isError ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
          }`}>
            {message.text}
          </div>
        )}

        {phase === 1 ? (
          /* Step 1: Email Form */
          <form onSubmit={handleRequestOTP} className="space-y-4">
            <p className="text-gray-600 text-sm text-center">
              Enter your registered email to receive a recovery code.
            </p>
            <div>
              <label className="block text-sm font-medium text-gray-700">Email Address</label>
              <input
                type="email"
                required
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-md transition duration-200 disabled:opacity-50"
            >
              {loading ? 'Sending...' : 'Send OTP'}
            </button>
          </form>
        ) : (
          /* Step 2: OTP & New Password Form */
          <form onSubmit={handleResetPassword} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">Enter OTP</label>
              <input
                type="text"
                required
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                value={otp}
                onChange={(e) => setOtp(e.target.value)}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">New Password</label>
              <input
                type="password"
                required
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              className="w-full py-2 px-4 bg-green-600 hover:bg-green-700 text-white font-semibold rounded-md transition duration-200 disabled:opacity-50"
            >
              {loading ? 'Verifying...' : 'Reset Password'}
            </button>
          </form>
        )}

        <div className="mt-6 text-center">
          <Link to="/login" className="text-sm text-blue-600 hover:underline">
            Back to Login
          </Link>
        </div>
      </div>
    </div>
  );
};

export default ForgotPassword;
