import unittest
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.hf_downloader import HFModelDownloader

class TestHFModelDownloader(unittest.TestCase):
    
    @patch("ml.hf_downloader.snapshot_download")
    def test_check_exists_success(self, mock_snapshot):
        # Simular que el modelo existe
        mock_snapshot.return_value = "cached/path"
        
        downloader = HFModelDownloader("fake/model")
        self.assertTrue(downloader.check_exists())
        
        status = downloader.get_status()
        self.assertTrue(status["is_downloaded"])
        self.assertEqual(status["progress"], 100)
        self.assertFalse(status["is_downloading"])
        
    @patch("ml.hf_downloader.snapshot_download")
    def test_check_exists_fails(self, mock_snapshot):
        # Simular que no existe localmente
        mock_snapshot.side_effect = Exception("Model not found in cache")
        
        downloader = HFModelDownloader("fake/model2")
        self.assertFalse(downloader.check_exists())
        
        status = downloader.get_status()
        self.assertFalse(status["is_downloaded"])
        
    @patch("ml.hf_downloader.snapshot_download")
    @patch("ml.hf_downloader.HfApi")
    def test_download_sync(self, mock_hf_api, mock_snapshot):
        # Mocks para simular la descarga sin red
        mock_api_instance = mock_hf_api.return_value
        
        mock_sibling = MagicMock()
        mock_sibling.size = 2048 # 2 KB
        mock_info = MagicMock()
        mock_info.siblings = [mock_sibling]
        
        mock_api_instance.model_info.return_value = mock_info
        
        downloader = HFModelDownloader("fake/model3")
        
        # Simular que el local_files_only=True falla (no cacheado) pero la descarga pasa
        def snapshot_side_effect(*args, **kwargs):
            if kwargs.get("local_files_only"):
                raise Exception("Not cached")
            return "fake/dir"
            
        mock_snapshot.side_effect = snapshot_side_effect
        
        downloader.download_sync()
        
        status = downloader.get_status()
        self.assertTrue(status["is_downloaded"])
        self.assertEqual(status["total_bytes"], 2048)
        self.assertEqual(status["downloaded_bytes"], 2048)
        self.assertEqual(status["progress"], 100)
        self.assertEqual(status["message"], "Modelo listo y cacheado.")
        self.assertTrue("speed_mbps" in status)

    @patch("ml.hf_downloader.snapshot_download")
    @patch("ml.hf_downloader.HfApi")
    def test_download_sync_network_error(self, mock_hf_api, mock_snapshot):
        # Mocks para simular fallo de red
        mock_api_instance = mock_hf_api.return_value
        mock_api_instance.model_info.side_effect = ConnectionError("Offline")
        
        downloader = HFModelDownloader("fake/model4")
        
        def snapshot_side_effect(*args, **kwargs):
            if kwargs.get("local_files_only"):
                raise Exception("Not cached")
            return "fake/dir"
        mock_snapshot.side_effect = snapshot_side_effect
        
        # Al ejecutar debe propagar o al menos setear el estado en error
        with self.assertRaises(ConnectionError):
            downloader.download_sync()
            
        status = downloader.get_status()
        self.assertFalse(status["is_downloading"])
        self.assertTrue("Error" in status["message"])
        
if __name__ == '__main__':
    unittest.main()
