CREATE EXTERNAL SCHEMA db_name_ext
FROM DATA CATALOG 
DATABASE 'db_name' 
IAM_ROLE '<IAM_ROLE_ARN>'
CREATE EXTERNAL DATABASE IF NOT EXISTS;


select * from db_name_ext.customers;
select * from db_name_ext.orders;


CREATE SCHEMA IF NOT EXISTS analytics;

CREATE TABLE IF NOT EXISTS analytics.top_customers (
    customer_id           VARCHAR(50),
    customer_name         VARCHAR(100),
    total_products_ordered INTEGER
)
DISTSTYLE KEY
DISTKEY(customer_id)
SORTKEY(total_products_ordered DESC);


insert into analytics.top_customers
with customer_orders as (
    select customer_id, count(product_id) as total_products_ordered
    from db_name_ext.orders
    group by customer_id
)
select co.customer_id, c.customer_name, co.total_products_ordered
from customer_orders co
join db_name_ext.customers c
on co.customer_id = c.customer_id
where co.total_products_ordered > 10
order by co.total_products_ordered desc
limit 10;

