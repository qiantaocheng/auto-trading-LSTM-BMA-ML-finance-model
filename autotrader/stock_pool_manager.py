#!/usr/bin/env python3
"""
股票池管理模块
提供股票池的创建、编辑、存储和查询功能
"""

import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class StockPoolManager:
    """股票池管理器"""
    
    def __init__(self, db_path: str = "trading_system.db"):
        """
        初始化股票池管理器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._init_database()
        self._load_default_pools()
    
    def _init_database(self):
        """初始化数据库表"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 创建股票池表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS stock_pools (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pool_name TEXT UNIQUE NOT NULL,
                        description TEXT,
                        tickers TEXT NOT NULL,
                        tags TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1,
                        metadata TEXT
                    )
                """)
                
                # 创建股票池历史记录表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS stock_pool_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pool_id INTEGER,
                        action TEXT,
                        old_tickers TEXT,
                        new_tickers TEXT,
                        changed_by TEXT,
                        changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (pool_id) REFERENCES stock_pools(id)
                    )
                """)
                
                # 创建索引
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_pool_name 
                    ON stock_pools(pool_name)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_pool_active 
                    ON stock_pools(is_active)
                """)
                
                conn.commit()
                logger.info("股票池数据库表初始化成功")
                
        except Exception as e:
            logger.error(f"初始化数据库失败: {e}")
            raise
    
    def _load_default_pools(self):
        """加载默认股票池"""
        default_pools = [
            {
                "name": "美股大盘蓝筹",
                "description": "美国大型蓝筹股，市值前30",
                "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", 
                           "BRK.B", "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA",
                           "DIS", "BAC", "ADBE", "NFLX", "CRM", "PFE", "TMO",
                           "ABT", "WMT", "CVX", "KO", "LLY", "NKE", "MCD", "VZ"],
                "tags": ["蓝筹", "大盘", "美股"]
            },
            {
                "name": "科技成长股",
                "description": "高成长科技股票组合",
                "tickers": ["NVDA", "AMD", "MSFT", "GOOGL", "META", "AAPL", "AMZN",
                           "CRM", "ADBE", "NOW", "SHOP", "SQ", "ROKU", "SNAP",
                           "UBER", "LYFT", "DOCU", "ZM", "OKTA", "TWLO"],
                "tags": ["科技", "成长", "高波动"]
            },
            {
                "name": "价值投资组合",
                "description": "低估值高股息股票",
                "tickers": ["BRK.B", "JPM", "BAC", "WFC", "GS", "MS", "C",
                           "JNJ", "PFE", "MRK", "ABBV", "CVS", "UNH",
                           "XOM", "CVX", "COP", "T", "VZ", "IBM", "INTC"],
                "tags": ["价值", "股息", "稳健"]
            },
            {
                "name": "ETF组合",
                "description": "主要指数ETF",
                "tickers": ["SPY", "QQQ", "DIA", "IWM", "VTI", "VOO",
                           "EEM", "EFA", "GLD", "SLV", "USO", "TLT",
                           "HYG", "LQD", "AGG", "VNQ", "XLF", "XLK"],
                "tags": ["ETF", "指数", "分散"]
            }
        ]
        
        for pool in default_pools:
            # 检查是否已存在
            if not self.get_pool_by_name(pool["name"]):
                self.create_pool(
                    pool_name=pool["name"],
                    tickers=pool["tickers"],
                    description=pool["description"],
                    tags=pool["tags"]
                )
    
    def create_pool(self, pool_name: str, tickers: List[str], 
                   description: str = "", tags: List[str] = None,
                   metadata: Dict = None) -> int:
        """
        创建新的股票池
        
        Args:
            pool_name: 股票池名称
            tickers: 股票代码列表
            description: 描述
            tags: 标签列表
            metadata: 元数据
            
        Returns:
            新创建的股票池ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                tickers_json = json.dumps(tickers)
                tags_json = json.dumps(tags) if tags else "[]"
                metadata_json = json.dumps(metadata) if metadata else "{}"
                
                cursor.execute("""
                    INSERT INTO stock_pools (pool_name, description, tickers, tags, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (pool_name, description, tickers_json, tags_json, metadata_json))
                
                pool_id = cursor.lastrowid
                
                # 记录历史
                cursor.execute("""
                    INSERT INTO stock_pool_history (pool_id, action, new_tickers)
                    VALUES (?, 'CREATE', ?)
                """, (pool_id, tickers_json))
                
                conn.commit()
                logger.info(f"创建股票池: {pool_name}, ID: {pool_id}")
                return pool_id
                
        except sqlite3.IntegrityError:
            logger.error(f"股票池名称已存在: {pool_name}")
            raise ValueError(f"股票池名称已存在: {pool_name}")
        except Exception as e:
            logger.error(f"创建股票池失败: {e}")
            raise
    
    def update_pool(self, pool_id: int, tickers: List[str] = None,
                   pool_name: str = None, description: str = None,
                   tags: List[str] = None) -> bool:
        """
        更新股票池
        
        Args:
            pool_id: 股票池ID
            tickers: 新的股票列表
            pool_name: 新名称
            description: 新描述
            tags: 新标签
            
        Returns:
            是否更新成功
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 获取旧数据
                old_pool = self.get_pool_by_id(pool_id)
                if not old_pool:
                    raise ValueError(f"股票池不存在: ID {pool_id}")
                
                updates = []
                params = []
                
                if pool_name is not None:
                    updates.append("pool_name = ?")
                    params.append(pool_name)
                
                if description is not None:
                    updates.append("description = ?")
                    params.append(description)
                
                if tickers is not None:
                    updates.append("tickers = ?")
                    params.append(json.dumps(tickers))
                    
                    # 记录历史
                    cursor.execute("""
                        INSERT INTO stock_pool_history (pool_id, action, old_tickers, new_tickers)
                        VALUES (?, 'UPDATE', ?, ?)
                    """, (pool_id, old_pool['tickers'], json.dumps(tickers)))
                
                if tags is not None:
                    updates.append("tags = ?")
                    params.append(json.dumps(tags))
                
                if updates:
                    updates.append("updated_at = CURRENT_TIMESTAMP")
                    params.append(pool_id)
                    
                    sql = f"UPDATE stock_pools SET {', '.join(updates)} WHERE id = ?"
                    cursor.execute(sql, params)
                    conn.commit()
                    
                    logger.info(f"更新股票池: ID {pool_id}")
                    return True
                    
                return False
                
        except Exception as e:
            logger.error(f"更新股票池失败: {e}")
            raise
    
    def delete_pool(self, pool_id: int) -> bool:
        """
        删除股票池（软删除）
        
        Args:
            pool_id: 股票池ID
            
        Returns:
            是否删除成功
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE stock_pools 
                    SET is_active = 0, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (pool_id,))
                
                # 记录历史
                cursor.execute("""
                    INSERT INTO stock_pool_history (pool_id, action)
                    VALUES (?, 'DELETE')
                """, (pool_id,))
                
                conn.commit()
                logger.info(f"删除股票池: ID {pool_id}")
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"删除股票池失败: {e}")
            raise
    
    def get_pool_by_id(self, pool_id: int) -> Optional[Dict]:
        """获取指定ID的股票池"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM stock_pools 
                    WHERE id = ? AND is_active = 1
                """, (pool_id,))
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_dict(row)
                return None
                
        except Exception as e:
            logger.error(f"获取股票池失败: {e}")
            return None
    
    def get_pool_by_name(self, pool_name: str) -> Optional[Dict]:
        """获取指定名称的股票池"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM stock_pools 
                    WHERE pool_name = ? AND is_active = 1
                """, (pool_name,))
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_dict(row)
                return None
                
        except Exception as e:
            logger.error(f"获取股票池失败: {e}")
            return None
    
    def get_all_pools(self, include_inactive: bool = False) -> List[Dict]:
        """
        获取所有股票池
        
        Args:
            include_inactive: 是否包含已删除的股票池
            
        Returns:
            股票池列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if include_inactive:
                    cursor.execute("SELECT * FROM stock_pools ORDER BY updated_at DESC")
                else:
                    cursor.execute("""
                        SELECT * FROM stock_pools 
                        WHERE is_active = 1 
                        ORDER BY updated_at DESC
                    """)
                
                rows = cursor.fetchall()
                return [self._row_to_dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"获取股票池列表失败: {e}")
            return []
    
    def add_tickers_to_pool(self, pool_id: int, tickers: List[str]) -> bool:
        """
        向股票池添加股票
        
        Args:
            pool_id: 股票池ID
            tickers: 要添加的股票列表
            
        Returns:
            是否添加成功
        """
        pool = self.get_pool_by_id(pool_id)
        if not pool:
            raise ValueError(f"股票池不存在: ID {pool_id}")
        
        current_tickers = json.loads(pool['tickers'])
        # 去重
        new_tickers = list(set(current_tickers + tickers))
        
        return self.update_pool(pool_id, tickers=new_tickers)
    
    def remove_tickers_from_pool(self, pool_id: int, tickers: List[str]) -> bool:
        """
        从股票池移除股票
        
        Args:
            pool_id: 股票池ID
            tickers: 要移除的股票列表
            
        Returns:
            是否移除成功
        """
        pool = self.get_pool_by_id(pool_id)
        if not pool:
            raise ValueError(f"股票池不存在: ID {pool_id}")
        
        current_tickers = json.loads(pool['tickers'])
        new_tickers = [t for t in current_tickers if t not in tickers]
        
        return self.update_pool(pool_id, tickers=new_tickers)
    
    def get_pool_tickers(self, pool_id: int = None, pool_name: str = None) -> List[str]:
        """
        获取股票池中的股票列表
        
        Args:
            pool_id: 股票池ID
            pool_name: 股票池名称
            
        Returns:
            股票代码列表
        """
        if pool_id:
            pool = self.get_pool_by_id(pool_id)
        elif pool_name:
            pool = self.get_pool_by_name(pool_name)
        else:
            raise ValueError("必须提供pool_id或pool_name")
        
        if pool:
            return json.loads(pool['tickers'])
        return []
    
    def search_pools(self, keyword: str = None, tag: str = None) -> List[Dict]:
        """
        搜索股票池
        
        Args:
            keyword: 搜索关键词（名称或描述）
            tag: 标签
            
        Returns:
            符合条件的股票池列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = "SELECT * FROM stock_pools WHERE is_active = 1"
                params = []
                
                if keyword:
                    query += " AND (pool_name LIKE ? OR description LIKE ?)"
                    params.extend([f"%{keyword}%", f"%{keyword}%"])
                
                if tag:
                    query += " AND tags LIKE ?"
                    params.append(f"%{tag}%")
                
                query += " ORDER BY updated_at DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [self._row_to_dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"搜索股票池失败: {e}")
            return []
    
    def get_pool_history(self, pool_id: int) -> List[Dict]:
        """
        获取股票池的历史记录
        
        Args:
            pool_id: 股票池ID
            
        Returns:
            历史记录列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM stock_pool_history 
                    WHERE pool_id = ? 
                    ORDER BY changed_at DESC
                """, (pool_id,))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"获取股票池历史失败: {e}")
            return []
    
    def _row_to_dict(self, row) -> Dict:
        """将数据库行转换为字典"""
        return dict(row)
    
    def export_pool(self, pool_id: int, filepath: str = None) -> str:
        """
        导出股票池到文件
        
        Args:
            pool_id: 股票池ID
            filepath: 导出文件路径
            
        Returns:
            导出文件路径
        """
        pool = self.get_pool_by_id(pool_id)
        if not pool:
            raise ValueError(f"股票池不存在: ID {pool_id}")
        
        if not filepath:
            filepath = f"stock_pool_{pool['pool_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'pool_name': pool['pool_name'],
            'description': pool['description'],
            'tickers': json.loads(pool['tickers']),
            'tags': json.loads(pool['tags']),
            'exported_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"导出股票池到: {filepath}")
        return filepath
    
    def import_pool(self, filepath: str, overwrite: bool = False) -> int:
        """
        从文件导入股票池
        
        Args:
            filepath: 导入文件路径
            overwrite: 是否覆盖同名股票池
            
        Returns:
            导入的股票池ID
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pool_name = data.get('pool_name')
        existing_pool = self.get_pool_by_name(pool_name)
        
        if existing_pool:
            if overwrite:
                return self.update_pool(
                    existing_pool['id'],
                    tickers=data.get('tickers'),
                    description=data.get('description'),
                    tags=data.get('tags')
                )
            else:
                # 生成新名称
                pool_name = f"{pool_name}_imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return self.create_pool(
            pool_name=pool_name,
            tickers=data.get('tickers', []),
            description=data.get('description', ''),
            tags=data.get('tags', [])
        )