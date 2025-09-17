#!/usr/bin/env python3
"""
股票池管理模块
提供股票池的创建、编辑、存储和查询功能
"""

import json
import os
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
        self.db_path = self._resolve_db_path(db_path)
        self._init_database()
        self._load_default_pools()
    
    @staticmethod
    def _sanitize_ticker(raw: str) -> Optional[str]:
        """规范化单个股票代码：去除空白和引号，统一大小写，保留常见符号。"""
        if raw is None:
            return None
        t = str(raw)
        # 去除中英文引号
        for ch in ['"', "'", '“', '”', '‘', '’']:
            t = t.replace(ch, '')
        # 统一空白
        t = t.replace('\u3000', ' ')
        t = t.strip()
        # 移除内部所有空白
        t = ''.join(c for c in t if not c.isspace())
        if not t:
            return None
        t = t.upper()
        # 统一分隔符（常见类股后缀）
        t = t.replace('/', '.')
        # XYZ-A -> XYZ.A（仅在后缀为单字符时）
        if '-' in t:
            parts = t.split('-')
            if len(parts) == 2 and len(parts[1]) == 1 and parts[1].isalpha():
                t = parts[0] + '.' + parts[1]
        # 仅保留允许字符
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")
        t = ''.join(ch for ch in t if ch in allowed)
        return t or None

    @classmethod
    def _sanitize_tickers(cls, tickers: List[str]) -> List[str]:
        """批量规范化股票代码，去重并保持顺序。"""
        sanitized: List[str] = []
        seen = set()
        for raw in tickers or []:
            st = cls._sanitize_ticker(raw)
            if st and st not in seen:
                sanitized.append(st)
                seen.add(st)
        return sanitized

    @staticmethod
    def _sanitize_pool_name(name: str) -> str:
        return (name or '').strip()

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

    def _resolve_db_path(self, raw_path: str) -> str:
        """将数据库路径解析到可写目录，避免权限导致的新建失败。"""
        try:
            p = Path(raw_path)
            if p.is_absolute():
                p.parent.mkdir(parents=True, exist_ok=True)
                return str(p)
            # 相对路径：尝试多个候选可写目录
            candidates = []
            module_dir = Path(__file__).resolve().parent
            candidates.append(module_dir / 'data')
            candidates.append(Path.cwd() / 'data')
            # 用户目录下的应用数据
            home_app = Path.home() / 'Autotrader' / 'data'
            candidates.append(home_app)
            for base in candidates:
                try:
                    base.mkdir(parents=True, exist_ok=True)
                    test_file = base / '.write_test'
                    with open(test_file, 'w', encoding='utf-8') as f:
                        f.write('ok')
                    os.remove(test_file)
                    dbp = base / raw_path
                    logger.info(f"股票池数据库路径: {dbp}")
                    return str(dbp)
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"数据库路径解析失败，将回退默认路径: {e}")
        # 最后回退当前目录
        return raw_path
    
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
            pool_name = self._sanitize_pool_name(pool_name)
            tickers = self._sanitize_tickers(tickers)
            tags = [str(t).strip() for t in (tags or []) if str(t).strip()]
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
                    params.append(self._sanitize_pool_name(pool_name))
                
                if description is not None:
                    updates.append("description = ?")
                    params.append(description)
                
                if tickers is not None:
                    norm_tickers = self._sanitize_tickers(tickers)
                    updates.append("tickers = ?")
                    params.append(json.dumps(norm_tickers))
                    
                    # 记录历史
                    cursor.execute("""
                        INSERT INTO stock_pool_history (pool_id, action, old_tickers, new_tickers)
                        VALUES (?, 'UPDATE', ?, ?)
                    """, (pool_id, old_pool['tickers'], json.dumps(norm_tickers)))
                
                if tags is not None:
                    clean_tags = [str(t).strip() for t in (tags or []) if str(t).strip()]
                    updates.append("tags = ?")
                    params.append(json.dumps(clean_tags))
                
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
        to_add = self._sanitize_tickers(tickers)
        # 合并去重并保持顺序
        merged = list(dict.fromkeys(current_tickers + to_add))
        
        return self.update_pool(pool_id, tickers=merged)
    
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