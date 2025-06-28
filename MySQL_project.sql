use coustumers_intension_db;

SELECT 
    visitorid,
    COUNT(CASE WHEN event = 'view' THEN 1 END) AS no_views,
    COUNT(CASE WHEN event = 'addtocart' THEN 1 END) AS no_addtocart,
    COUNT(CASE WHEN event = 'transaction' THEN 1 END) AS no_transaction,
    (MAX(timestamp) - MIN(timestamp)) / (1000 * 60) AS time_spent_minutes
    
FROM 
    events
GROUP BY 
    visitorid
ORDER BY 
    visitorid;

